"""Appearance augmentation: video in, relit video out.

Drives MIMIC's relighting pipeline (:mod:`lav_randomize_video`, in this folder)
with the prompt store in :mod:`mimic.appearance_augmentation.prompts`. The
diffusion backbone comes from the Light-A-Video submodule under
``third_party/Light-A-Video``, which is used unmodified.

The output video keeps the input's frame count, resolution and fps, so an
augmented clip stays frame-aligned with the action labels of the original.

The video is split into fixed-length segments; each segment is relit under one
prompt sampled from the pool, so a single long clip sweeps several lighting
conditions. ``--n_lights 1`` (or ``--prompt``) keeps one condition throughout.

Usage:
    # one video, 4 lighting conditions sampled from the full pool
    python -m mimic.appearance_augmentation.augment_video \\
        --input clip.mp4 --output clip_aug.mp4 --n_lights 4 --seed 42

    # restrict to night and rain conditions
    python -m mimic.appearance_augmentation.augment_video \\
        --input clip.mp4 --output clip_aug.mp4 --categories night rain

    # one explicit prompt for the whole clip
    python -m mimic.appearance_augmentation.augment_video \\
        --input clip.mp4 --output clip_aug.mp4 --prompt "sidewalk scene, dense fog at dusk"

    # show the plan without loading models (no GPU needed)
    python -m mimic.appearance_augmentation.augment_video \\
        --input clip.mp4 --output clip_aug.mp4 --dry_run

Requires the Light-A-Video runtime (torch, diffusers, ultralytics, …) and a CUDA
GPU: ``uv sync --extra appearance``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

from . import prompts as prompt_store

#: This package — holds MIMIC's own lav_*.py pipeline scripts.
PACKAGE_DIR = Path(__file__).resolve().parent

#: Light-A-Video submodule (upstream code, unmodified).
DEFAULT_LAV_ROOT = PACKAGE_DIR / "third_party" / "Light-A-Video"

#: Where IC-Light and YOLO weights are cached. Both download on first use.
DEFAULT_MODELS_DIR = PACKAGE_DIR / "models"

#: Seconds of video relit under a single sampled prompt.
DEFAULT_SEGMENT_SEC = 8

_IC_LIGHT_WEIGHTS = "iclight_sd15_fc.safetensors"


def resolve_lav_root(explicit: str | None = None) -> Path:
    """Locate the Light-A-Video checkout: ``--lav_root``, then ``$LAV_ROOT``, then the submodule."""
    for candidate in (explicit, os.environ.get("LAV_ROOT"), DEFAULT_LAV_ROOT):
        if not candidate:
            continue
        root = Path(candidate).resolve()
        # src/ic_light.py is upstream's, so it marks a real checkout rather than an empty dir.
        if (root / "src" / "ic_light.py").is_file():
            return root
    raise FileNotFoundError(
        f"Light-A-Video not found at {DEFAULT_LAV_ROOT}. The submodule is probably "
        "not initialized — run:\n"
        "  git submodule update --init --recursive\n"
        "Or point at an existing checkout with --lav_root / $LAV_ROOT."
    )


def _import_lav(lav_root: Path):
    """Import MIMIC's pipeline module with the Light-A-Video submodule importable.

    Two entries go on ``sys.path``: this package (for ``lav_randomize_video`` and
    its sibling ``lav_wan_sidewalk``) and the submodule root (for upstream's
    ``src.*`` and ``utils.*``). The process working directory is left alone —
    every model path handed to the pipeline is absolute.
    """
    for entry in (str(PACKAGE_DIR), str(lav_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import lav_randomize_video  # our pipeline; resolved from PACKAGE_DIR

    return lav_randomize_video


def video_seed(base_seed: int, video_path: str) -> int:
    """Deterministic per-video seed, so different clips draw different prompts."""
    digest = int(hashlib.sha256(str(video_path).encode()).hexdigest(), 16)
    return (base_seed ^ digest) & 0xFFFF_FFFF


def build_pipeline_args(
    seed: int = 42,
    n_lights: int = 4,
    max_side: int = 480,
    strength: float = 0.35,
    num_step: int = 10,
    text_guide_scale: float = 1.5,
    gamma: float = 0.7,
    fg_preserve: float = 0.3,
    detail_strength: float = 0.7,
    no_yolo: bool = False,
    upscaler: str = "none",
    compile_vdm: bool = False,
    models_dir: Path | None = None,
) -> SimpleNamespace:
    """Build the config object consumed by ``lav_randomize_video.load_pipeline``.

    Model paths are absolute so the pipeline does not depend on the process cwd.
    """
    models = Path(models_dir or DEFAULT_MODELS_DIR)
    models.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        seed=seed,
        n_lights=n_lights,
        max_side=max_side,
        strength=strength,
        num_step=num_step,
        text_guide_scale=text_guide_scale,
        gamma=gamma,
        fg_preserve=0.0 if no_yolo else fg_preserve,
        detail_strength=detail_strength,
        no_yolo=no_yolo,
        yolo_model=str(models / "yolov8n.pt"),
        upscaler=upscaler,
        negative_prompt=prompt_store.NEGATIVE_PROMPT,
        sd_model="stablediffusionapi/realistic-vision-v51",
        vdm_model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ic_light_model=str(models / _IC_LIGHT_WEIGHTS),
        compile=compile_vdm,
    )


def apply_prompt_pool(lav_module, pool: list[str], segment_sec: int = DEFAULT_SEGMENT_SEC) -> None:
    """Point the pipeline at the MIMIC prompt store.

    ``process_one_video`` reads its prompt pool, light directions and segment
    length from module globals, so they are rebound here rather than passed in.
    """
    lav_module.RELIGHT_PROMPT_POOL = pool
    lav_module.BG_SOURCES = prompt_store.BG_SOURCES
    lav_module.BG_WEIGHTS = prompt_store.BG_WEIGHTS
    lav_module.SEGMENT_DURATION_SEC = segment_sec


def augment_video(state, lav_module, input_path: Path, output_path: Path) -> bool:
    """Relight one video. ``state`` comes from ``load_pipeline``; paths must be absolute.

    Load the pipeline once and call this repeatedly to batch many videos.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return lav_module.process_one_video(state, str(input_path), str(output_path))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Appearance augmentation for sidewalk navigation video: video in, relit video out.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--input", type=str, required=True, help="Input video (.mp4).")
    io_group.add_argument(
        "--output", type=str, default=None,
        help="Output video. Default: alongside the input as <stem>_aug.mp4.",
    )
    io_group.add_argument(
        "--lav_root", type=str, default=None,
        help="Light-A-Video checkout. Default: the third_party submodule, or $LAV_ROOT.",
    )
    io_group.add_argument(
        "--models_dir", type=str, default=None,
        help=f"Weight cache for IC-Light and YOLO. Default: {DEFAULT_MODELS_DIR}",
    )
    io_group.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    io_group.add_argument("--dry_run", action="store_true", help="Print the plan and exit without loading models.")

    prompt_group = parser.add_argument_group("prompt selection")
    prompt_group.add_argument(
        "--prompt", type=str, default=None,
        help="Use one explicit prompt for the whole video, bypassing the pool.",
    )
    prompt_group.add_argument(
        "--categories", nargs="+", default=None,
        help=f"Restrict the pool to these categories. Available: {', '.join(prompt_store.CATEGORIES)}",
    )
    prompt_group.add_argument("--no_simulator", action="store_true", help="Exclude simulator-style prompts.")
    prompt_group.add_argument("--n_lights", type=int, default=4, help="Distinct lighting conditions sampled per video.")
    prompt_group.add_argument("--segment_sec", type=int, default=DEFAULT_SEGMENT_SEC, help="Seconds of video per lighting condition.")
    prompt_group.add_argument("--seed", type=int, default=42)
    prompt_group.add_argument(
        "--fixed_seed", action="store_true",
        help="Use --seed directly instead of mixing in the input path (which varies prompts across clips).",
    )

    quality = parser.add_argument_group("quality / speed")
    quality.add_argument("--fast", action="store_true", help="Lower resolution and fewer steps.")
    quality.add_argument("--max_side", type=int, default=None, help="Max side of the model resolution. Default: 480 (512 with --fast).")
    quality.add_argument("--strength", type=float, default=None, help="Relight strength. Default: 0.35 (0.25 with --fast).")
    quality.add_argument("--num_step", type=int, default=None, help="Step scale; actual steps = round(num_step/strength). Default: 10 (6 with --fast).")
    quality.add_argument("--text_guide_scale", type=float, default=1.5)
    quality.add_argument("--gamma", type=float, default=0.7, help="Consistent-light-attention mixing weight.")
    quality.add_argument("--fg_preserve", type=float, default=0.3, help="How strongly detected people are kept from the original frames.")
    quality.add_argument("--detail_strength", type=float, default=0.7, help="How much high-frequency detail is carried over from the original.")
    quality.add_argument("--no_yolo", action="store_true", help="Skip person detection (faster; relights the full frame).")
    quality.add_argument("--upscaler", type=str, default="none", choices=["none", "realesrgan", "sd_x4"])
    quality.add_argument("--compile", action="store_true", dest="compile_vdm", help="torch.compile the video diffusion model.")

    args = parser.parse_args(argv)

    if args.fast:
        defaults = {"max_side": 512, "strength": 0.25, "num_step": 6}
    else:
        defaults = {"max_side": 480, "strength": 0.35, "num_step": 10}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input video not found: {input_path}", file=sys.stderr)
        return 1
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_aug.mp4")
    )
    if output_path == input_path:
        print("error: --output would overwrite the input video", file=sys.stderr)
        return 1
    if output_path.exists() and not args.overwrite:
        print(f"error: {output_path} exists (pass --overwrite to replace it)", file=sys.stderr)
        return 1

    if args.prompt:
        pool = [args.prompt]
        n_lights = 1
    else:
        try:
            pool = prompt_store.build_prompt_pool(
                args.categories, include_simulator=not args.no_simulator
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        n_lights = args.n_lights

    seed = args.seed if args.fixed_seed else video_seed(args.seed, str(input_path))

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Prompts: {len(pool)} in pool, {min(n_lights, len(pool))} sampled per video, {args.segment_sec}s each")
    print(f"Seed:    {seed}" + ("" if args.fixed_seed else f" (derived from {args.seed} + input path)"))
    # Mirrors the draw inside process_one_video, so this preview is what actually runs.
    for prompt in random.Random(seed).sample(pool, min(n_lights, len(pool))):
        print(f"  - {prompt}")

    if args.dry_run:
        print("\nDry run: models not loaded, no output written.")
        return 0

    try:
        lav_root = resolve_lav_root(args.lav_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"\nLight-A-Video: {lav_root}")

    lav = _import_lav(lav_root)
    apply_prompt_pool(lav, pool, segment_sec=args.segment_sec)

    pipeline_args = build_pipeline_args(
        seed=seed,
        n_lights=n_lights,
        max_side=args.max_side,
        strength=args.strength,
        num_step=args.num_step,
        text_guide_scale=args.text_guide_scale,
        gamma=args.gamma,
        fg_preserve=args.fg_preserve,
        detail_strength=args.detail_strength,
        no_yolo=args.no_yolo,
        upscaler=args.upscaler,
        compile_vdm=args.compile_vdm,
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )

    state = lav.load_pipeline(pipeline_args)
    state.seed = seed
    state.vdm_prompt = prompt_store.VDM_PROMPT

    ok = augment_video(state, lav, input_path, output_path)
    if not ok:
        print("error: no frames produced", file=sys.stderr)
        return 1
    print(f"Done! Saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
