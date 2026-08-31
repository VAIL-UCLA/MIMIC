# Appearance Augmentation

Re-renders a sidewalk navigation video under new lighting, weather and
time-of-day conditions. Geometry, frame count, resolution, fps and the ego
trajectory are preserved, so an augmented clip stays frame-aligned with the
action labels of the original — only appearance changes.

Backed by [Light-A-Video](https://github.com/bcmi/Light-A-Video) (ICCV 2025):
IC-Light supplies the per-frame relighting, a Wan2.1 video diffusion model holds
temporal consistency, and Consistent Light Attention plus Progressive Light
Fusion suppress flicker.

## Layout

| Path | Purpose |
|---|---|
| `Light-A-Video` | Symlink to the Light-A-Video checkout (gitignored — see below) |
| `prompts.py` | Prompt store: 171 lighting prompts in 15 categories, plus light directions |
| `augment_video.py` | CLI and API — one video in, one relit video out |

## Setup

The `Light-A-Video` symlink is machine-specific and not committed. Recreate it:

```bash
ln -sfn /path/to/Light-A-Video mimic/appearance_augmentation/Light-A-Video
```

Alternatively set `$LAV_ROOT` or pass `--lav_root`. The checkout must contain
`models/iclight_sd15_fc.safetensors`; the SD, AnimateDiff and Wan2.1 weights
download automatically on first run.

Runtime (CUDA GPU required):

```bash
uv sync --extra appearance
```

Or run inside Light-A-Video's own `lav` conda environment.

## Usage

```bash
# 4 lighting conditions sampled from the full pool, 8s of video each
python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --output clip_aug.mp4 --n_lights 4 --seed 42

# restrict to night and rain conditions
python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --categories night rain --n_lights 3

# one explicit prompt for the whole clip
python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --prompt "sidewalk scene, dense fog at dusk"

# preview the plan without loading models (no GPU needed)
python -m mimic.appearance_augmentation.augment_video --input clip.mp4 --dry_run

# faster, lower fidelity
python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --fast --no_yolo
```

Default output is `<input_stem>_aug.mp4` beside the input.

## How prompts are chosen

The video is split into `--segment_sec` (default 8s) segments. `--n_lights`
distinct prompts are sampled from the pool, and segment *i* uses prompt
*i % n_lights* — so a long clip sweeps several conditions while short clips get
one. `--n_lights 1` or `--prompt` keeps a single condition throughout.

The sampling seed is derived from `--seed` **and the input path**, so different
clips in a batch draw different conditions from the same `--seed`. Pass
`--fixed_seed` to use `--seed` verbatim.

Inspect the prompt store directly:

```bash
python -m mimic.appearance_augmentation.prompts --list
python -m mimic.appearance_augmentation.prompts --categories night rain --sample 5
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

lav_root = resolve_lav_root()
lav = _import_lav(lav_root)                       # NOTE: chdir's into lav_root
apply_prompt_pool(lav, build_prompt_pool(["night", "rain"]))
state = lav.load_pipeline(build_pipeline_args(lav_root=lav_root))

for src, dst in clips:                            # absolute Path objects
    state.seed = video_seed(42, str(src))
    augment_video(state, lav, src, dst)
```

`_import_lav` changes the process working directory into the Light-A-Video
checkout, because upstream resolves weights against relative paths. Absolutize
your own paths first.

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
