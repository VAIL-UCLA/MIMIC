#!/usr/bin/env python
"""Stage 2 — action augmentation over a corpus of clips.

Recorded data only shows the robot driving well, so a policy trained on it has
never seen itself off-course. This manufactures that experience: for each clip
the robot drifts laterally off the recorded path and corrects back onto it, and
both the re-rendered video and the matching action label are generated.

    # labels only -- no GPU, no models, fast enough to run over a whole corpus
    uv run python scripts/stage_2_action_augmentation.py \\
        --input assets/clips --output out/action --labels_only

    # two maneuvers per clip, rendered
    uv run python scripts/stage_2_action_augmentation.py \\
        --input assets/clips --output out/action --variants 2 --fps 20

    # see what would be generated
    uv run python scripts/stage_2_action_augmentation.py \\
        --input assets/clips --dry_run

Rendering needs a per-clip depth scale, since the depth reconstruction is not
metric and its units depend on the clip. Clips missing a ``.scale.json`` are
calibrated first unless ``--no_calibrate`` is given; a clip that cannot be
calibrated is reported rather than rendered at a guessed scale.

Each clip is rendered by a fresh TrajectoryCrafter, which reloads its weights —
upstream binds the model to one clip at construction. ``--labels_only`` skips
all of it and is the right mode when only the labels are wanted.

Needs the action extra and a CUDA GPU to render: ``uv sync --extra action``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stage_common import (
    banner,
    clip_label,
    find_clips,
    human_time,
    progress,
    say,
    step,
    summarize,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from mimic.action_augmentation import augment_action as aa
from mimic.action_augmentation import labels as label_io
from mimic.action_augmentation import scales as scale_io
from mimic.action_augmentation import trajectory as tj

STAGE = "Stage 2"
TITLE = "Action augmentation"


def variant_name(video: Path, index: int, variants: int) -> str:
    stem = video.parent.name if video.stem.startswith("rgb_") else video.stem
    return f"{stem}_act.mp4" if variants == 1 else f"{stem}_act{index}.mp4"


def _last_error(output: str, code: int) -> str:
    """Pull the actionable line out of a captured failure."""
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.lower().startswith("error:"):
            return line[len("error:"):].strip()
    return lines[-1] if lines else f"exit code {code}"


def planned_strength(video: Path, seed: int, args) -> tuple[int, float]:
    """The seed and offset this clip will actually get.

    Mirrors what ``augment_action`` samples, so the plan printed here and the
    run that follows cannot drift apart.
    """
    clip_seed = seed if args.fixed_seed else aa.clip_seed(seed, str(video))
    strength = aa.sample_strength(
        clip_seed, args.strength,
        tuple(args.strength_range) if args.strength_range else None,
    )
    return clip_seed, strength


def describe(job: dict, args) -> str:
    side = "left" if job["strength"] >= 0 else "right"
    return (f"{job['strength']:+.2f} m {side}, {args.horizon:.0f}s from "
            f"t={job['start_time']:.1f}s")


def ensure_scale(video: Path, args, bar) -> tuple[float | None, str]:
    """Resolve the clip's depth scale, calibrating it first if needed and allowed."""
    window = scale_io.make_window(aa.TC_VIDEO_LENGTH, args.stride)
    try:
        return scale_io.resolve_scale(video, args.scale, window=window)
    except scale_io.ScaleNotFound:
        pass

    if args.no_calibrate:
        return None, ("no .scale.json and --no_calibrate is set; run "
                      "scripts/../calibrate_clips or pass --scale")

    from mimic.action_augmentation import calibrate_clips as cc

    say(bar, f"    calibrating depth scale for {clip_label(video)} ...")
    argv = ["--input", str(video), "--length", str(aa.TC_VIDEO_LENGTH),
            "--stride", str(args.stride), "--device", args.device]
    if args.fps:
        argv += ["--fps", str(args.fps)]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = cc.main(argv)
    if code != 0:
        tail = buffer.getvalue().strip().splitlines()
        return None, f"calibration failed: {tail[-1] if tail else 'unknown error'}"
    return scale_io.resolve_scale(video, args.scale, window=window)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 2: generate deviate-and-recover maneuvers across a corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--input", required=True,
                          help="Clip directory, video file, or glob (quote it).")
    io_group.add_argument("--output", default="out/action",
                          help="Directory for the generated clips and labels.")
    io_group.add_argument("--stream", default="rgb_pinhole.mp4",
                          help="Which stream to take from each clip folder.")
    io_group.add_argument("--variants", type=int, default=1,
                          help="Maneuvers to generate per clip, each differently sampled.")
    io_group.add_argument("--labels_only", action="store_true",
                          help="Trajectories and labels only, no video synthesis (no GPU).")
    io_group.add_argument("--overwrite", action="store_true")
    io_group.add_argument("--dry_run", action="store_true",
                          help="Print the plan; write nothing, load nothing.")
    io_group.add_argument("--manifest", default=None,
                          help="Where to write the run manifest. Default: <output>/manifest.json")
    io_group.add_argument("--fps", type=float, default=None,
                          help="Frame rate for sidecars without timestamps.")

    man = parser.add_argument_group("maneuver")
    man.add_argument("--strength", type=float, default=None,
                     help="Peak lateral offset in meters. Positive left, negative right.")
    man.add_argument("--strength_range", type=float, nargs=2, default=None,
                     metavar=("LO", "HI"),
                     help="Sample |offset| from [LO, HI] and coin-flip the side.")
    man.add_argument("--horizon", type=float, default=tj.DEFAULT_HORIZON_S,
                     help="Maneuver duration in seconds; the peak lands at half of it.")
    man.add_argument("--start_time", type=float, default=None,
                     help="When the maneuver begins, in clip time. Default: the "
                          "earliest window where the robot is actually moving — a "
                          "deviation from a parked robot teaches nothing.")
    man.add_argument("--min_speed", type=float, default=tj.DEFAULT_MIN_SPEED_MPS,
                     help="Speed above which the robot counts as moving.")
    man.add_argument("--min_moving", type=float, default=0.6,
                     help="Refuse a clip whose best window is less moving than this.")
    man.add_argument("--profile", default="raised_cosine", choices=tj.PROFILES)
    man.add_argument("--mode", default="deviate_recover",
                     choices=["deviate_recover", "reexpress"])
    man.add_argument("--seed", type=int, default=42)
    man.add_argument("--fixed_seed", action="store_true")

    render = parser.add_argument_group("rendering")
    render.add_argument("--scale", default=scale_io.AUTO,
                        help="Depth units per meter, or 'auto' to use each clip's sidecar.")
    render.add_argument("--no_calibrate", action="store_true",
                        help="Do not calibrate clips that lack a scale; fail them instead.")
    render.add_argument("--stride", type=int, default=1)
    render.add_argument("--diffusion_inference_steps", type=int, default=50)
    render.add_argument("--tc_root", default=None)
    render.add_argument("--device", default="cuda:0")

    args = parser.parse_args(argv)
    if args.strength is None and args.strength_range is None:
        args.strength_range = [0.3, 0.8]
    return args


def maneuver_window(video: Path, args) -> tuple[float, float]:
    """Where in this clip to put the maneuver, and how moving that window is."""
    sidecar = label_io.find_sidecar(video)
    data = label_io.load_labels(sidecar, fps=args.fps or label_io.DEFAULT_FPS)
    if args.start_time is not None:
        window = (data.times >= args.start_time) & (
            data.times <= args.start_time + args.horizon
        )
        speed = tj.path_speed(data.poses, data.times)
        moving = float((speed[window] > args.min_speed).mean()) if window.any() else 0.0
        return float(args.start_time), moving
    return tj.best_maneuver_start(data.poses, data.times, args.horizon, args.min_speed)


def build_argv(video: Path, target: Path, seed: int, scale, start_time: float,
               args) -> list[str]:
    argv = [
        "--input", str(video),
        "--output", str(target),
        "--output_labels", str(target.with_suffix(".npz")),
        "--horizon", str(args.horizon),
        "--start_time", str(start_time),
        "--profile", args.profile,
        "--mode", args.mode,
        "--seed", str(seed),
        "--stride", str(args.stride),
        "--diffusion_inference_steps", str(args.diffusion_inference_steps),
        "--device", args.device,
    ]
    if args.fixed_seed:
        argv.append("--fixed_seed")
    if args.fps:
        argv += ["--fps", str(args.fps)]
    if args.strength is not None:
        argv += ["--strength", str(args.strength)]
    elif args.strength_range:
        argv += ["--strength_range", str(args.strength_range[0]), str(args.strength_range[1])]
    if args.overwrite:
        argv.append("--overwrite")
    if args.labels_only:
        argv.append("--labels_only")
    else:
        argv += ["--scale", str(scale)]
    if args.tc_root:
        argv += ["--tc_root", args.tc_root]
    return argv


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    clips = find_clips(args.input, args.stream)
    if not clips:
        print(f"error: no clips matched {args.input!r}", file=sys.stderr)
        return 1

    out_dir = Path(args.output).expanduser().resolve()

    jobs, unusable = [], []
    for clip in clips:
        try:
            start, moving = maneuver_window(clip, args)
        except (FileNotFoundError, ValueError) as exc:
            unusable.append({"clip": clip_label(clip), "status": "failed",
                             "error": str(exc)})
            continue
        if moving < args.min_moving:
            unusable.append({
                "clip": clip_label(clip), "status": "failed",
                "error": f"best {args.horizon:.0f}s window is only {moving:.0%} moving "
                         f"(need {args.min_moving:.0%}); the robot is parked for most "
                         f"of this clip",
            })
            continue
        for index in range(args.variants):
            seed = args.seed + index
            clip_seed, strength = planned_strength(clip, seed, args)
            jobs.append({"clip": clip, "index": index, "seed": seed,
                         "clip_seed": clip_seed, "strength": strength,
                         "start_time": start, "moving": moving,
                         "target": out_dir / variant_name(clip, index, args.variants)})

    if not jobs:
        banner(STAGE, TITLE, {"clips": f"{len(clips)} from {args.input}",
                              "usable": "none"})
        for record in unusable:
            print(f"  ! {record['clip']}: {record['error']}")
        return 1

    offsets = np.array([j["strength"] for j in jobs])
    starts = np.array([j["start_time"] for j in jobs])
    placement = ("fixed at --start_time" if args.start_time is not None
                 else f"auto, t={starts.min():.1f} .. {starts.max():.1f}s "
                      f"(first window with the robot moving)")
    banner(STAGE, TITLE, {
        "clips": f"{len(clips)} from {args.input}"
                 + (f"  ({len(unusable)} unusable)" if unusable else ""),
        "variants": f"{args.variants} per clip  ({len(jobs)} maneuvers)",
        "maneuver": f"{args.mode}, {args.horizon:.0f}s horizon, {args.profile} profile",
        "placement": placement,
        "offsets": f"{offsets.min():+.2f} .. {offsets.max():+.2f} m  "
                   f"({(offsets > 0).sum()} left, {(offsets < 0).sum()} right)",
        "video": "skipped (--labels_only)" if args.labels_only
                 else f"TrajectoryCrafter, {args.diffusion_inference_steps} steps",
        "output": out_dir,
    })
    for record in unusable:
        print(f"  ! {record['clip']}: {record['error']}")
    if unusable:
        print()

    if args.dry_run:
        for job in jobs:
            print(f"  {clip_label(job['clip'])} -> {job['target'].name}   "
                  f"{describe(job, args)}   {job['moving']:.0%} moving   "
                  f"seed {job['clip_seed']}")
        print(f"\nDry run: {len(jobs)} maneuver(s) planned, nothing written.")
        return 0

    records = list(unusable)
    bar = progress(jobs, "Generating", unit="maneuver")
    for job in bar:
        clip, target = job["clip"], job["target"]
        name = clip_label(clip)
        record = {"clip": name, "output": str(target), "variant": job["index"],
                  "seed": job["clip_seed"], "strength_m": round(job["strength"], 4),
                  "mode": args.mode}

        label_target = target.with_suffix(".npz")
        done = label_target.exists() and (args.labels_only or target.exists())
        if done and not args.overwrite:
            record["status"] = "skipped"
            records.append(record)
            say(bar, f"  - {target.stem}  exists, skipped")
            continue

        scale = None
        if not args.labels_only:
            scale, provenance = ensure_scale(clip, args, bar)
            if scale is None:
                record["status"] = "failed"
                record["error"] = provenance
                say(bar, f"  ✗ {target.stem}  {provenance}")
                records.append(record)
                continue
            record["scale"] = scale
            record["scale_source"] = provenance

        with step(bar, f"Generating {name} · {describe(job, args)}"):
            item_started = time.perf_counter()
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    code = aa.main(build_argv(clip, target, job["seed"], scale,
                                              job["start_time"], args))
                record["status"] = "written" if code == 0 else "failed"
                if code != 0:
                    record["error"] = _last_error(buffer.getvalue(), code)
            except Exception as exc:  # noqa: BLE001 - one bad clip must not sink the corpus
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["seconds"] = round(time.perf_counter() - item_started, 1)

        mark = "✓" if record["status"] == "written" else "✗"
        detail = record.get(
            "error",
            f"{describe(job, args)}  [{human_time(record['seconds'])}]",
        )
        say(bar, f"  {mark} {label_target.name}  {detail}")
        records.append(record)
    bar.close()

    manifest = write_manifest(
        Path(args.manifest) if args.manifest else out_dir / "manifest.json",
        "action_augmentation",
        {"input": args.input, "variants": args.variants, "mode": args.mode,
         "horizon_s": args.horizon, "profile": args.profile,
         "strength": args.strength, "strength_range": args.strength_range,
         "seed": args.seed, "labels_only": args.labels_only},
        records,
    )
    return summarize(STAGE, records, time.perf_counter() - started, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
