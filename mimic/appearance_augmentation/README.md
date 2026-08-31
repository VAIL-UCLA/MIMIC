# Appearance Augmentation

Re-renders a sidewalk navigation video under new lighting, weather and
time-of-day conditions. Geometry, frame count, resolution, fps and the ego
trajectory are preserved, so an augmented clip stays frame-aligned with the
action labels of the original — only appearance changes.

The diffusion backbone is [Light-A-Video](https://github.com/bcmi/Light-A-Video)
(ICCV 2025), included as a **git submodule under `third_party/` and used
unmodified**. Everything else in this folder is MIMIC's own.

## What is ours vs. upstream

Upstream supplies the relighting core: IC-Light per-frame relighting, the
Wan2.1 / AnimateDiff video backbones, Consistent Light Attention and Progressive
Light Fusion. We did not change any of it — the submodule is pinned at upstream
`main` with a byte-identical tree.

Upstream's entry points, however, are built for **short single-shot demo clips**:
one hand-written prompt per YAML config, a center-crop to a fixed resolution, and
an 8 fps write. For navigation training data that is not usable — cropping breaks
the correspondence between frames and action labels, and one prompt per clip
gives no diversity. The scripts in this folder replace that entry layer.

| | Official Light-A-Video | MIMIC |
|---|---|---|
| **Entry point** | `lav_relight.py`, `lav_wan_relight.py` — one YAML per clip | `augment_video.py` CLI, plus batch and multi-GPU drivers |
| **Prompting** | one hand-written `relight_prompt` per config | 171-prompt pool in `prompts.py`, 15 selectable categories, sampled per segment |
| **Clip length** | one fixed-length clip (`num_frames` in config) | arbitrary length — 8 s segments split into 81-frame Wan chunks, padded and reassembled |
| **Conditions per clip** | a single lighting condition | `--n_lights` conditions swept across segments |
| **Resolution** | `resize_and_center_crop` to config `width`/`height` — **crops the frame** | native resolution preserved; the model runs at `--max_side` and the result is resized back |
| **Frame rate** | hardcoded `fps=8` on write | source fps preserved |
| **People in frame** | none in the relight path (SAM2 only in the separate inpainting demo) | YOLOv8 person detection → dilated, blurred soft masks → `--fg_preserve` blend toward the original, with an OpenCV-cascade fallback and a no-mask fallback |
| **Texture** | none | high-frequency residual from the original re-added after relighting (`--detail_strength`) |
| **Upscaling** | none | optional Real-ESRGAN or SD ×4 |
| **Batching** | one process per config file | model loaded once for N videos; `infer_all.sh` fans out across GPUs |
| **Seeding** | fixed per config | derived from `seed ⊕ hash(video path)`, so a batch under one seed still varies per clip |

### Why YOLO

Relighting a pedestrian tends to smear or deform them — the model happily
re-renders a person as a lit blob. `detect_faces_yolo` finds person boxes per
frame, dilates them ~30 px, blurs the mask, and `blend_with_original` pulls those
regions back toward the original pixels by `--fg_preserve`. Detection failure is
non-fatal: it falls back to an OpenCV Haar cascade, then to full-frame relighting.

`--no_yolo` skips detection entirely — noticeably faster, and fine for clips
with no people.

## Layout

| Path | Origin | Purpose |
|---|---|---|
| `third_party/Light-A-Video` | upstream, unmodified | Relighting core (submodule, pinned) |
| `prompts.py` | ours | 171 lighting prompts in 15 categories, light directions, negative prompt |
| `augment_video.py` | ours | CLI and API — one video in, one relit video out |
| `lav_randomize_video.py` | ours | Pipeline: segment/chunk scheduling, YOLO masks, blending, reassembly |
| `lav_wan_sidewalk.py` | ours | Wan2.1 sidewalk variant; supplies the upscaler helpers |
| `lav_paint_sidewalk.py` | ours | Inpainting variant for masked-region relighting |
| `lav_wan_baseline.py` | ours | Unmodified-baseline runs for comparison |
| `lav_batch_variants.py` | ours | Batch driver — load once, process a `[start:end]` slice |
| `extract_foreground.py` | ours | SAM2 foreground extraction |
| `infer_all.sh` | ours | Multi-GPU fan-out |

## Setup

```bash
git submodule update --init --recursive   # if you did not clone with --recurse-submodules
uv sync --extra appearance                # torch, diffusers, ultralytics — needs a CUDA GPU
```

Weights land in `models/` (gitignored) and download on first run: IC-Light
(`iclight_sd15_fc.safetensors`, ~1.7 GB) and YOLOv8n. The SD, AnimateDiff and
Wan2.1 checkpoints are pulled from HuggingFace automatically.

To use a Light-A-Video checkout you already have, pass `--lav_root` or set
`$LAV_ROOT` instead of initializing the submodule.

## Usage

```bash
# 4 lighting conditions sampled from the full pool, 8s of video each
uv run python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --output clip_aug.mp4 --n_lights 4 --seed 42

# restrict to night and rain conditions
uv run python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --categories night rain --n_lights 3

# one explicit prompt for the whole clip
uv run python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --prompt "sidewalk scene, dense fog at dusk"

# preview the plan without loading models (no GPU needed)
uv run python -m mimic.appearance_augmentation.augment_video --input clip.mp4 --dry_run

# faster, lower fidelity
uv run python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --fast --no_yolo
```

Default output is `<input_stem>_aug.mp4` beside the input.

## How prompts are chosen

The video is split into `--segment_sec` (default 8 s) segments. `--n_lights`
distinct prompts are sampled from the pool, and segment *i* uses prompt
*i % n_lights* — so a long clip sweeps several conditions while short clips get
one. `--n_lights 1` or `--prompt` keeps a single condition throughout.

The sampling seed is derived from `--seed` **and the input path**, so different
clips in a batch draw different conditions from the same `--seed`. Pass
`--fixed_seed` to use `--seed` verbatim.

Inspect the prompt store directly:

```bash
uv run python -m mimic.appearance_augmentation.prompts --list
uv run python -m mimic.appearance_augmentation.prompts --categories night rain --sample 5
```

Categories: `daytime_sun`, `overcast`, `seasons`, `urban_shadows`, `twilight`,
`night`, `rain`, `snow`, `fog`, `dust`, `covered`, `camera`, `geographic`,
`mixed`, `simulator`.

## Batching

Model loading dominates runtime, so load once and loop:

```python
from mimic.appearance_augmentation import build_prompt_pool
from mimic.appearance_augmentation.augment_video import (
    _import_lav, apply_prompt_pool, augment_video, build_pipeline_args,
    resolve_lav_root, video_seed,
)

lav = _import_lav(resolve_lav_root())
apply_prompt_pool(lav, build_prompt_pool(["night", "rain"]))
state = lav.load_pipeline(build_pipeline_args())

for src, dst in clips:                    # absolute Path objects
    state.seed = video_seed(42, str(src))
    augment_video(state, lav, src, dst)
```

Every model path is absolute, so this does not depend on the working directory.

For a large corpus laid out as `{root}/{scenario}/front_*.mp4`, the slice-based
driver and the GPU fan-out are simpler:

```bash
uv run python mimic/appearance_augmentation/lav_batch_variants.py \
    --root /path/to/5HZ_288H_512W --start_idx 0 --end_idx 250
bash mimic/appearance_augmentation/infer_all.sh <input_dir> <output_dir>
```

## Notable knobs

| Flag | Default | Effect |
|---|---|---|
| `--strength` | 0.35 | How far the relight departs from the original |
| `--num_step` | 10 | Step scale; actual steps = `round(num_step / strength)` |
| `--gamma` | 0.7 | Consistent Light Attention mixing weight — higher is smoother, flatter |
| `--fg_preserve` | 0.3 | How strongly detected people are kept from the original frames |
| `--detail_strength` | 0.7 | High-frequency detail carried over from the original |
| `--no_yolo` | off | Skip person detection; faster, relights the full frame |
| `--fast` | off | max_side 512, strength 0.25, 6 steps |
