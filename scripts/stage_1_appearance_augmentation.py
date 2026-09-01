#!/usr/bin/env python
"""Stage 1 — appearance augmentation over a corpus of clips.

Re-renders every clip under new lighting, weather and time-of-day conditions,
producing ``--variants`` differently-lit copies of each. Geometry, frame count,
resolution and fps are preserved, so an augmented clip stays frame-aligned with
the action labels of the original — only appearance changes, which is what lets
stage 2 reuse the same labels.

The relighting pipeline is loaded once and reused across the whole corpus;
running the per-clip CLI in a loop would pay the model load every time.

    uv run python scripts/stage_1_appearance_augmentation.py \\
        --input assets/clips --output out/appearance --variants 2

    # see the plan and the prompts each clip would draw, without a GPU
    uv run python scripts/stage_1_appearance_augmentation.py \\
        --input assets/clips --dry_run

Needs the appearance extra and a CUDA GPU: ``uv sync --extra appearance``.
"""

from __future__ import annotations

import argparse
import random
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

from mimic.appearance_augmentation import augment_video as av
from mimic.appearance_augmentation import prompts as prompt_store

STAGE = "Stage 1"
TITLE = "Appearance augmentation"


def variant_name(video: Path, index: int, variants: int) -> str:
    """Output filename for one relit copy of a clip."""
    stem = video.parent.name if video.stem.startswith("rgb_") else video.stem
    return f"{stem}.mp4" if variants == 1 else f"{stem}_light{index}.mp4"


def describe(prompts: list[str], limit: int = 2) -> str:
    """Condense sampled prompts into something that fits on the progress line."""
    cleaned = [p.replace(prompt_store.SCENE_PREFIX, "").strip() for p in prompts]
    short = [c.split(",")[0] for c in cleaned]
    if len(short) > limit:
        return ", ".join(short[:limit]) + f" +{len(short) - limit}"
    return ", ".join(short)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 1: relight a corpus of clips into new conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = parser.add_argument_group("input/output")
    io.add_argument("--input", required=True,
                    help="Clip directory, video file, or glob (quote it).")
    io.add_argument("--output", default="out/appearance",
                    help="Directory for the relit clips.")
    io.add_argument("--stream", default="rgb_pinhole.mp4",
                    help="Which stream to take from each clip folder.")
    io.add_argument("--variants", type=int, default=1,
                    help="Relit copies to generate per clip, each differently lit.")
    io.add_argument("--overwrite", action="store_true", help="Redo clips already written.")
    io.add_argument("--dry_run", action="store_true",
                    help="Print the plan and the sampled prompts; load no models.")
    io.add_argument("--manifest", default=None,
                    help="Where to write the run manifest. Default: <output>/manifest.json")

    sel = parser.add_argument_group("prompt selection")
    sel.add_argument("--categories", nargs="*", default=None,
                     choices=sorted(prompt_store.CATEGORIES),
                     metavar="NAME",
                     help="Restrict the pool to these categories. Default: all of "
                          + ", ".join(sorted(prompt_store.CATEGORIES)))
    sel.add_argument("--no_simulator", action="store_true",
                     help="Exclude simulator-style prompts.")
    sel.add_argument("--n_lights", type=int, default=4,
                     help="Lighting conditions sampled within one output video.")
    sel.add_argument("--segment_sec", type=int, default=av.DEFAULT_SEGMENT_SEC,
                     help="Seconds of video per lighting condition.")
    sel.add_argument("--seed", type=int, default=42)

    q = parser.add_argument_group("quality / speed")
    q.add_argument("--low_gpu_memory_mode", action="store_true",
                   help="Offload the video model between calls. The relighting "
                        "UNet stays resident, so holding both needs more than a "
                        "16 GB card has.")
    q.add_argument("--fast", action="store_true", help="Lower resolution and fewer steps.")
    q.add_argument("--max_side", type=int, default=None)
    q.add_argument("--strength", type=float, default=None)
    q.add_argument("--num_step", type=int, default=None)
    q.add_argument("--fg_preserve", type=float, default=0.3,
                   help="How strongly detected people are kept from the original frames.")
    q.add_argument("--detail_strength", type=float, default=0.7)
    q.add_argument("--no_yolo", action="store_true",
                   help="Skip person detection (faster; relights the full frame).")
    q.add_argument("--lav_root", default=None,
                   help="Light-A-Video checkout. Default: the third_party submodule.")
    q.add_argument("--models_dir", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    clips = find_clips(args.input, args.stream)
    if not clips:
        print(f"error: no clips matched {args.input!r}", file=sys.stderr)
        return 1

    try:
        pool = prompt_store.build_prompt_pool(
            args.categories, include_simulator=not args.no_simulator
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.output).expanduser().resolve()
    n_sampled = min(args.n_lights, len(pool))

    # Plan every output up front so the bar has a true total and the dry run
    # shows exactly what a real run would produce.
    jobs = []
    for clip in clips:
        for index in range(args.variants):
            target = out_dir / variant_name(clip, index, args.variants)
            seed = av.video_seed(args.seed + index, str(clip))
            drawn = random.Random(seed).sample(pool, n_sampled)
            jobs.append({"clip": clip, "index": index, "target": target,
                         "seed": seed, "prompts": drawn})

    banner(STAGE, TITLE, {
        "clips": f"{len(clips)} from {args.input}",
        "variants": f"{args.variants} per clip  ({len(jobs)} videos to generate)",
        "prompts": f"{len(pool)} in pool, {n_sampled} conditions per video, "
                   f"{args.segment_sec}s each",
        "people": "preserved via YOLO" if not args.no_yolo else "not preserved (--no_yolo)",
        "output": out_dir,
    })

    if args.dry_run:
        for job in jobs:
            print(f"  {clip_label(job['clip'])} -> {job['target'].name}   seed {job['seed']}")
            for prompt in job["prompts"]:
                print(f"      · {prompt}")
        print(f"\nDry run: {len(jobs)} video(s) planned, no models loaded, nothing written.")
        return 0

    try:
        lav_root = av.resolve_lav_root(args.lav_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Loading Light-A-Video from {lav_root} ...", flush=True)
    load_started = time.perf_counter()
    lav = av._import_lav(lav_root)
    av.apply_prompt_pool(lav, pool, segment_sec=args.segment_sec)
    pipeline_args = av.build_pipeline_args(
        seed=args.seed, n_lights=n_sampled,
        max_side=args.max_side or (512 if args.fast else 480),
        strength=args.strength if args.strength is not None else (0.25 if args.fast else 0.35),
        num_step=args.num_step if args.num_step is not None else (6 if args.fast else 10),
        fg_preserve=args.fg_preserve, detail_strength=args.detail_strength,
        no_yolo=args.no_yolo,
        low_gpu_memory_mode=args.low_gpu_memory_mode,
        vdm_prompt=prompt_store.VDM_PROMPT,
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )
    state = lav.load_pipeline(pipeline_args)
    state.vdm_prompt = prompt_store.VDM_PROMPT
    print(f"  ready in {human_time(time.perf_counter() - load_started)}\n")

    records = []
    bar = progress(jobs, "Relighting", unit="video")
    for job in bar:
        clip, target = job["clip"], job["target"]
        name = clip_label(clip)
        record = {"clip": name, "output": str(target), "variant": job["index"],
                  "seed": job["seed"], "prompts": job["prompts"]}

        if target.exists() and not args.overwrite:
            record["status"] = "skipped"
            records.append(record)
            say(bar, f"  - {target.name}  exists, skipped")
            continue

        conditions = describe(job["prompts"])
        with step(bar, f"Generating {name} v{job['index']} · {conditions}"):
            item_started = time.perf_counter()
            try:
                state.seed = job["seed"]
                ok = av.augment_video(state, lav, clip, target)
                record["status"] = "written" if ok else "failed"
                if not ok:
                    record["error"] = "pipeline produced no frames"
            except Exception as exc:  # noqa: BLE001 - one bad clip must not sink the corpus
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["seconds"] = round(time.perf_counter() - item_started, 1)

        mark = "✓" if record["status"] == "written" else "✗"
        detail = record.get("error", f"{conditions}  [{human_time(record['seconds'])}]")
        say(bar, f"  {mark} {target.name}  {detail}")
        records.append(record)
    bar.close()

    manifest = write_manifest(
        Path(args.manifest) if args.manifest else out_dir / "manifest.json",
        "appearance_augmentation",
        {"input": args.input, "variants": args.variants, "n_lights": args.n_lights,
         "segment_sec": args.segment_sec, "categories": args.categories,
         "seed": args.seed, "no_yolo": args.no_yolo},
        records,
    )
    return summarize(STAGE, records, time.perf_counter() - started, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
