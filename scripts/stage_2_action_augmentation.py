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
import subprocess
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
from mimic.action_augmentation import windows as win

STAGE = "Stage 2"
TITLE = "Action augmentation"


def variant_name(video: Path, index: int, variants: int) -> str:
    stem = video.parent.name if video.stem.startswith("rgb_") else video.stem
    return f"{stem}_act.mp4" if variants == 1 else f"{stem}_act{index}.mp4"


#: Upstream writes the refined clip under this name inside its output folder.
RENDER_NAME = "gen.mp4"


def render_output(target: Path) -> Path:
    """Where the rendered clip for ``target`` actually lands.

    ``--output x_act.mp4`` does not produce that file: upstream treats the stem
    as a folder and writes ``x_act/gen.mp4`` alongside its intermediates. The
    "already done" check has to look at what is really written, or every clip
    re-renders and then fails because its labels are already there.
    """
    return target.parent / target.stem / RENDER_NAME


def _last_error(output: str, code: int) -> str:
    """Pull the actionable line out of a captured failure."""
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.lower().startswith("error:"):
            return line[len("error:"):].strip()
    return lines[-1] if lines else f"exit code {code}"


def planned_strength(video: Path, seed: int, speed: float, args) -> tuple[int, float]:
    """The seed and offset this clip will actually get.

    Mirrors what ``augment_action`` samples, so the plan printed here and the
    run that follows cannot drift apart.
    """
    clip_seed = seed if args.fixed_seed else aa.clip_seed(seed, str(video))
    strength = aa.sample_strength(
        clip_seed, args.strength,
        tuple(args.strength_range) if args.strength_range else None,
        args.strength_scale, speed,
    )
    return clip_seed, strength


def describe(job: dict, args) -> str:
    side = "left" if job["strength"] >= 0 else "right"
    return (f"{job['strength']:+.2f} m {side} ({job['yaw_deg']:.0f} deg), "
            f"{args.horizon:.0f}s from t={job['start_time']:.1f}s")


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
            "--stride", str(args.stride), "--device", args.device,
            "--cpu_offload", args.cpu_offload]
    if args.fps:
        argv += ["--fps", str(args.fps)]
    if args.in_process:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = cc.main(argv)
        output = buffer.getvalue()
    else:
        # Same reason as the render: a CUDA context left in this process is
        # memory the renderer's subprocess does not get.
        completed = subprocess.run(
            [sys.executable, "-m", "mimic.action_augmentation.calibrate_clips", *argv],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        )
        code, output = completed.returncode, completed.stdout + completed.stderr
    if code != 0:
        tail = output.strip().splitlines()
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
    io_group.add_argument("--oom_retries", type=int, default=1,
                          help="Retries when a clip fails with CUDA out-of-memory. "
                               "The peak sits close to a 16 GB card's capacity, so "
                               "whether it fits depends on what else is resident.")
    io_group.add_argument("--in_process", action="store_true",
                          help="Render inside this interpreter instead of one "
                               "subprocess per clip. Easier to debug, but one "
                               "clip's leftover GPU memory can fail the next.")
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
                     help="Sample |offset| from [LO, HI] meters and coin-flip the side.")
    man.add_argument("--strength_scale", type=float, default=None,
                     help="Size the offset as scale * the clip's top speed in the "
                          "maneuver window, scale in seconds. This is the default "
                          f"({tj.DEFAULT_STRENGTH_SCALE_S}s): a fixed metric offset asks "
                          "the same swerve of a robot crawling at 0.3 m/s as of one at "
                          "3 m/s, and the slow one would have to turn 58 degrees to do it.")
    man.add_argument("--horizon", type=float, default=tj.DEFAULT_HORIZON_S,
                     help="Maneuver duration in seconds; the peak lands at half of it.")
    man.add_argument("--start_time", type=float, default=None,
                     help="When the maneuver begins, in clip time. Default: the "
                          "earliest window where the robot is actually moving — a "
                          "deviation from a parked robot teaches nothing.")
    man.add_argument("--min_speed", type=float, default=tj.DEFAULT_MIN_SPEED_MPS,
                     help="Speed above which the robot counts as moving.")
    man.add_argument("--no_auto_window", action="store_true",
                     help="Do not move the render window onto the maneuver. The "
                          "renderer only reads the leading frames, so a maneuver "
                          "later in the clip is simply not rendered.")
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
    render.add_argument("--cpu_offload", default="model",
                        choices=("model", "sequential"),
                        help="DepthCrafter offload strategy.")
    render.add_argument("--prompt", default=None,
                        help="Scene description for the refinement model, shared by "
                             "every clip. Default: upstream captions each clip with "
                             "BLIP-2, which pins ~7.5 GB to the GPU. Supplying one "
                             "frees that, and is effectively required below ~24 GB.")
    render.add_argument("--low_gpu_memory_mode", action="store_true",
                        help="Upstream's low-VRAM path for the refinement "
                             "model. Needed below roughly 24 GB; slower.")

    args = parser.parse_args(argv)
    given = [n for n, v in (("--strength", args.strength),
                            ("--strength_range", args.strength_range),
                            ("--strength_scale", args.strength_scale)) if v is not None]
    if len(given) > 1:
        parser.error(f"{' and '.join(given)} are mutually exclusive")
    if not given:
        args.strength_scale = tj.DEFAULT_STRENGTH_SCALE_S
    return args


def maneuver_window(video: Path, args) -> tuple[float, float, float]:
    """Where in this clip to put the maneuver, how usable it is, and how fast.

    The speed is the maximum over the chosen window, which is what
    ``--strength_scale`` sizes the offset against.
    """
    sidecar = label_io.find_sidecar(video)
    fps = args.fps or label_io.clip_fps(video) or label_io.DEFAULT_FPS
    data = label_io.load_labels(sidecar, fps=fps)
    speeds = tj.path_speed(data.poses, data.times)

    if args.start_time is not None:
        start = float(args.start_time)
        mask = (data.times >= start) & (data.times <= start + args.horizon)
        usable = float((speeds[mask] > args.min_speed).mean()) if mask.any() else 0.0
    else:
        start, usable = tj.best_maneuver_start(
            data.poses, data.times, args.horizon, args.min_speed
        )
        mask = (data.times >= start) & (data.times <= start + args.horizon)

    speed = float(speeds[mask].max()) if mask.any() else float(speeds.max(initial=0.0))
    return start, usable, speed


def run_augment(argv: list[str], args) -> tuple[int, str]:
    """Render one clip, by default in a subprocess.

    Rendering in-process leaves each clip at the mercy of what the previous one
    left on the card: the same clip succeeds or OOMs depending on its position
    in the run, because freeing a pipeline does not always return every block to
    the allocator. A subprocess gives each clip the whole GPU in a known state,
    and costs one model load per clip.

    ``--in_process`` restores the old behaviour for debugging, where a traceback
    in the same interpreter is easier to work with.
    """
    if args.in_process:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = aa.main(argv)
        return code, buffer.getvalue()

    command = [sys.executable, "-m", "mimic.action_augmentation.augment_action", *argv]
    for attempt in range(args.oom_retries + 1):
        completed = subprocess.run(
            command, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode == 0 or "OutOfMemoryError" not in output:
            return completed.returncode, output
        if attempt < args.oom_retries:
            # The peak lands within a few hundred MB of the card's capacity, so
            # whether it fits depends on what else is on the GPU right now — a
            # desktop compositor is enough to decide it. Retrying costs a model
            # load and usually finds the memory.
            _free_gpu()
    return completed.returncode, output


def _free_gpu() -> None:
    """Best-effort wait for another process to hand memory back."""
    time.sleep(15)


def window_start(video: Path, start_time: float, args) -> int:
    """First source frame of the window the renderer should see, 0 when unmoved.

    The renderer only reads the leading frames, so a maneuver later in the clip
    is invisible to it unless the window is moved onto it.
    """
    if args.no_auto_window:
        return 0
    fps = args.fps or label_io.clip_fps(video) or label_io.DEFAULT_FPS
    data = label_io.load_labels(label_io.find_sidecar(video), fps=fps)
    return win.window_start_for(
        start_time, args.horizon, fps, aa.TC_VIDEO_LENGTH, args.stride, len(data.poses)
    )


def prepare_clip(job, args, bar) -> tuple[Path, float]:
    """The clip the renderer should be pointed at, and the maneuver time in it.

    Returns the clip unchanged when the maneuver already sits in the leading
    window; otherwise materializes that stretch as its own bundle so it does.
    """
    offset = job.get("window_start", 0)
    if not offset:
        return job["clip"], job["start_time"]

    fps = args.fps or label_io.clip_fps(job["clip"]) or label_io.DEFAULT_FPS
    dest = (Path(args.output).expanduser().resolve()
            / "_windows" / job["clip"].parent.name)
    say(bar, f"    windowing {clip_label(job['clip'])} from frame {offset} ...")
    video = win.materialize(
        job["clip"], dest, offset, aa.TC_VIDEO_LENGTH, args.stride, fps=fps
    )
    return video, max(0.0, job["start_time"] - offset / fps)


def build_argv(video: Path, target: Path, seed: int, scale, start_time: float,
               strength: float, args) -> list[str]:
    """Argv for one maneuver.

    The offset is passed explicitly rather than re-sampled downstream, so the
    plan printed here and the run that follows cannot disagree.
    """
    argv = [
        "--input", str(video),
        "--output", str(target),
        "--output_labels", str(target.with_suffix(".npz")),
        "--strength", str(strength),
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
    if args.overwrite:
        argv.append("--overwrite")
    if args.labels_only:
        argv.append("--labels_only")
    else:
        argv += ["--scale", str(scale)]
    if args.tc_root:
        argv += ["--tc_root", args.tc_root]
    if args.cpu_offload:
        argv += ["--cpu_offload", args.cpu_offload]
    if args.prompt:
        argv += ["--prompt", args.prompt]
    if args.low_gpu_memory_mode:
        argv.append("--low_gpu_memory_mode")
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
            start, moving, speed = maneuver_window(clip, args)
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
            clip_seed, strength = planned_strength(clip, args.seed, speed, args)
            # Variants alternate sides. Under --strength_scale the magnitude is
            # fixed by the clip's speed, so the side is the only thing left to
            # vary — and drawing it independently would give two variants the
            # same side half the time.
            if index % 2:
                strength = -strength
            jobs.append({"clip": clip, "index": index, "seed": seed,
                         "clip_seed": clip_seed, "strength": strength,
                         "start_time": start, "moving": moving, "speed": speed,
                         "window_start": window_start(clip, start, args),
                         "yaw_deg": np.degrees(tj.peak_yaw(strength, args.horizon, speed)),
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
    yaws = np.array([j["yaw_deg"] for j in jobs])
    sizing = (f"{args.strength_scale}s x window top speed "
              f"({np.array([j['speed'] for j in jobs]).min():.1f}"
              f"-{np.array([j['speed'] for j in jobs]).max():.1f} m/s)"
              if args.strength_scale is not None else "absolute meters")
    banner(STAGE, TITLE, {
        "clips": f"{len(clips)} from {args.input}"
                 + (f"  ({len(unusable)} unusable)" if unusable else ""),
        "variants": f"{args.variants} per clip  ({len(jobs)} maneuvers)",
        "maneuver": f"{args.mode}, {args.horizon:.0f}s horizon, {args.profile} profile",
        "placement": placement,
        "sizing": sizing,
        "offsets": f"{offsets.min():+.2f} .. {offsets.max():+.2f} m  "
                   f"({(offsets > 0).sum()} left, {(offsets < 0).sum()} right)",
        "peak yaw": f"{yaws.min():.1f} .. {yaws.max():.1f} deg",
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
        done = label_target.exists() and (
            args.labels_only or render_output(target).exists()
        )
        if done and not args.overwrite:
            record["status"] = "skipped"
            records.append(record)
            say(bar, f"  - {target.stem}  exists, skipped")
            continue

        # The renderer reads the leading frames only, so bring the maneuver
        # there first: calibration and rendering must both see that window.
        render_clip, render_start = clip, job["start_time"]
        if not args.labels_only:
            try:
                render_clip, render_start = prepare_clip(job, args, bar)
            except (OSError, ValueError) as exc:
                record["status"] = "failed"
                record["error"] = f"windowing failed: {exc}"
                say(bar, f"  ✗ {target.stem}  {record['error']}")
                records.append(record)
                continue
            if render_clip != clip:
                record["window_start"] = job["window_start"]

        scale = None
        if not args.labels_only:
            scale, provenance = ensure_scale(render_clip, args, bar)
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
            argv = build_argv(render_clip, target, job["seed"], scale,
                              render_start, job["strength"], args)
            try:
                code, output = run_augment(argv, args)
                record["status"] = "written" if code == 0 else "failed"
                if code != 0:
                    record["error"] = _last_error(output, code)
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
