#!/usr/bin/env python3
"""Batch variant generation: load model once, process many videos with start_idx/end_idx for parallel runs.

Discovers front_*.mp4 under root/{scenario}/, sorts them, and processes videos[start_idx:end_idx].
Output: same directory as each input, with name front_*_var.mp4.

Each video gets a **unique seed** derived from (base_seed XOR hash(video_path)) so every
video samples different lighting prompts from the pool.

Example layout:
  Root: /path/to/dataset/5HZ_288H_512W
  Input:  {root}/{scenario}/front_*.mp4
  Output: {root}/{scenario}/front_*_var.mp4

Usage:
  # List and sort all videos (no processing)
  python lav_batch_variants.py --root /path/to/5HZ_288H_512W --list_only

  # Process all videos (single process)
  python lav_batch_variants.py --root /path/to/5HZ_288H_512W

  # Process slice for parallel runs (e.g. 4 workers: 0-250, 250-500, 500-750, 750-1000)
  python lav_batch_variants.py --root /path/to/5HZ_288H_512W --start_idx 0   --end_idx 250
  python lav_batch_variants.py --root /path/to/5HZ_288H_512W --start_idx 250 --end_idx 500

  # Skip YOLO person detection for speed (full frame relighting, no foreground preservation)
  python lav_batch_variants.py --root /path/to/5HZ_288H_512W --no_yolo
"""

import os
import sys
import argparse
import glob
import hashlib

# Run from Light-A-Video directory so imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from lav_randomize_video import load_pipeline, process_one_video


def _video_seed(base_seed: int, video_path: str) -> int:
    """Deterministic per-video seed so every video gets different prompts."""
    h = int(hashlib.sha256(video_path.encode()).hexdigest(), 16)
    return (base_seed ^ h) & 0xFFFF_FFFF


def discover_videos(root: str, pattern: str = "front_*.mp4") -> list[str]:
    """Find all videos matching {root}/*/pattern. Sort for deterministic order."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        scenario_dir = os.path.join(root, name)
        if not os.path.isdir(scenario_dir):
            continue
        for path in sorted(glob.glob(os.path.join(scenario_dir, pattern))):
            if os.path.isfile(path):
                out.append(path)
    return out


def output_path_for(video_path: str) -> str:
    """Same directory as video, stem + _var.mp4. E.g. .../front_abc.mp4 -> .../front_abc_var.mp4"""
    d = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}_var.mp4")


def main():
    parser = argparse.ArgumentParser(
        description="Batch relight variants: load model once, process videos[start_idx:end_idx].",
    )
    parser.add_argument("--root", type=str, required=True,
                        help="Root dir containing scenario subdirs with front_*.mp4 (e.g. .../5HZ_288H_512W)")
    parser.add_argument("--pattern", type=str, default="front_*.mp4",
                        help="Glob pattern under each scenario dir (default: front_*.mp4)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index into sorted video list (for parallel splits)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (exclusive). Default: process all from start_idx")
    parser.add_argument("--list_only", action="store_true",
                        help="Only list sorted video paths and exit")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos that already have a _var.mp4 output (default: True)")
    parser.add_argument("--no_yolo", action="store_true",
                        help="Skip YOLO person detection (much faster, full-frame relighting)")
    parser.add_argument("--n_lights", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max_side", type=int, default=None)
    parser.add_argument("--compile", action="store_true", dest="compile_", help="Use torch.compile on VDM")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    args = parser.parse_args()

    videos = discover_videos(args.root, args.pattern)
    videos = sorted(videos)
    n_total = len(videos)
    print(f"Found {n_total} videos under {args.root} (sorted)")

    if args.list_only:
        for i, p in enumerate(videos):
            print(f"  {i}: {p}")
        return

    end = args.end_idx if args.end_idx is not None else n_total
    start = max(0, args.start_idx)
    end = min(end, n_total)
    slice_list = videos[start:end]
    print(f"Processing slice [{start}:{end}] → {len(slice_list)} videos")

    if not slice_list:
        print("No videos in slice.")
        return

    class PipelineArgs:
        pass
    pargs = PipelineArgs()
    pargs.max_side = args.max_side or (512 if args.fast else 480)
    pargs.n_lights = args.n_lights
    pargs.seed = args.seed
    pargs.yolo_model = "yolov8n.pt"
    pargs.fg_preserve = 0.0 if args.no_yolo else 0.3
    pargs.detail_strength = 0.7
    pargs.upscaler = "none"
    pargs.negative_prompt = (
        "bad quality, worse quality, low quality, low resolution, blurry, blur, out of focus, "
        "distorted, deformed, disfigured, ugly, bad anatomy, wrong proportions, "
        "oversaturated, underexposed, overexposed, flat lighting, harsh shadows, "
        "artifacts, noise, grain, watermark, text, logo, duplicate, mutilated, "
        "cropped, jpeg artifacts, compression artifacts, flickering, inconsistent lighting"
    )
    pargs.strength = 0.25 if args.fast else 0.35
    pargs.num_step = 6 if args.fast else 10
    pargs.text_guide_scale = 1.5
    pargs.gamma = 0.7
    pargs.sd_model = "stablediffusionapi/realistic-vision-v51"
    pargs.vdm_model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    pargs.ic_light_model = os.path.join(SCRIPT_DIR, "models", "iclight_sd15_fc.safetensors")
    pargs.compile = getattr(args, "compile_", False)
    pargs.no_yolo = args.no_yolo

    state = load_pipeline(pargs)

    done = 0
    skipped = 0
    failed = 0
    for i, video_path in enumerate(slice_list):
        out_path = output_path_for(video_path)
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            print(f"  [{start + i + 1}/{end}] skip (exists): {os.path.basename(out_path)}")
            continue

        state.seed = _video_seed(args.seed, video_path)

        print(f"  [{start + i + 1}/{end}] {os.path.basename(video_path)} → {os.path.basename(out_path)}"
              f"  (seed={state.seed})")
        try:
            ok = process_one_video(state, video_path, out_path)
            if ok:
                done += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"    ERROR: {e}")

    if state.upscaler_obj is not None:
        del state.upscaler_obj
    import torch
    torch.cuda.empty_cache()

    print(f"\nDone: {done} written, {skipped} skipped (existing), {failed} failed.")


if __name__ == "__main__":
    main()
