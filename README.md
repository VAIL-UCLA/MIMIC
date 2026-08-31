# MIMIC

Official implementation of **"Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion"** (ICRA 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2603.22527-blue)](https://arxiv.org/abs/2603.22527)
[![ICRA](https://img.shields.io/badge/ICRA-2026-orange)](https://arxiv.org/abs/2603.22527)
[![Model Zoo](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Zoo-yellow)](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public)

MIMIC is a **goal-free, long-context** sidewalk navigation policy: it takes a short history of RGB frames and predicts a local trajectory for autonomous sidewalk driving.

## Installation

This repository is managed with [uv](https://docs.astral.sh/uv/). Install uv, then:

```bash
git clone --recurse-submodules https://github.com/VAIL-UCLA/MIMIC.git
cd MIMIC
uv sync
```

Already cloned without `--recurse-submodules`? Run
`git submodule update --init --recursive`.

`uv sync` creates `.venv` on CPython 3.11.11 (pinned in `.python-version`) and
installs `mimic` in editable mode. Run anything with `uv run`:

```bash
uv run python -m mimic.appearance_augmentation.prompts --list
```

Optional extras:

| Extra | Install | For |
|---|---|---|
| `appearance` | `uv sync --extra appearance` | Appearance augmentation (torch, diffusers, ultralytics — needs a CUDA GPU) |
| `action` | `uv sync --extra action` | Action augmentation video synthesis (needs a CUDA GPU; the label path needs neither) |
| `dev` | `uv sync --extra dev` | ruff, pytest |

If you only want to run inference with the released ONNX policy, you can skip
uv entirely — see [Pretrained Model](#pretrained-model) below.

## Pretrained Model

The pretrained MIMIC policy is released in the **[UCLA-VAIL/Navigation-Model-Zoo-Public](https://huggingface.co/UCLA-VAIL/Navigation-Model-Zoo-Public)** model zoo — exported to ONNX (`mimic.onnx`) behind a unified inference interface.

| Property | Value |
|---|---|
| Goal mode | goal-free |
| Context | 16 RGB frames |
| Input resolution | 288 × 512 (fixed — frames must already be this size; **not** resized internally) |
| Normalization | none — pixel values in `[0, 1]` |
| Input rate | 5 Hz |
| Output | 15 waypoints `(x, y, yaw)` at non-uniform timestamps 0.2 s–5.0 s; the wrapper keeps the first 13 (~4 s) |
| Frame | standard: `x = forward`, `y = left`, meters |

### 1. Download

```bash
pip install -U "huggingface_hub[cli]"
hf download UCLA-VAIL/Navigation-Model-Zoo-Public --include "MIMIC/*" --local-dir ./nav_model_zoo
```

### 2. Dependencies

```bash
pip install onnxruntime-gpu numpy torch pyyaml   # use onnxruntime instead for CPU-only
```

> ⚠️ **`urbansim` is required to import.** `MIMIC/inference.py` runs `from urbansim.custom.pp import PurePursuitController` at module load, so the `urbansim` package must be importable. It is only used to construct a helper that the shipped inference path never calls — if you don't have `urbansim`, comment out that import to run pure ONNX inference.

### 3. Run inference

```python
import numpy as np
from MIMIC.inference import MIMICNavigator      # run from ./nav_model_zoo

nav = MIMICNavigator(device="cuda")             # device="cpu" if no GPU

# obs: the robot's last 16 RGB frames at 288×512, (1, 16, 3, 288, 512) float32 in [0, 1]
obs = np.random.rand(1, 16, 3, 288, 512).astype(np.float32)

# MIMIC is goal-free — no goal argument
traj, scores = nav.inference_trajectory(obs)    # (1, 1, 13, 2) meters
vw, best     = nav.inference_vw(obs)            # vw: (1, 2) = [v, ω];  best: (1, 13, 2)
nav.reset()                                     # clear PD smoothing between episodes
```

`inference_trajectory` returns local waypoints in meters; `inference_vw` turns the trajectory into a `[linear_v, angular_ω]` command via a built-in PD controller (tune limits with `max_v` / `max_w` at construction). Feed frames in temporal order `[t-15, …, t]`.

## Data Augmentation

The `mimic/` package holds the corrective behavior expansion pipeline used to
grow the training set from recorded sidewalk footage.

| Module | Purpose |
|---|---|
| [`mimic/appearance_augmentation`](mimic/appearance_augmentation) | Re-render a clip under new lighting, weather and time-of-day conditions |
| [`mimic/action_augmentation`](mimic/action_augmentation) | Generate deviate-and-recover maneuvers with matching action labels |

### Appearance augmentation

Re-renders a clip under new lighting, weather and time-of-day conditions.
Geometry, frame count, resolution and fps are preserved, so an augmented clip
stays frame-aligned with the action labels of the original — only appearance
changes.

The relighting core is [Light-A-Video](https://github.com/bcmi/Light-A-Video)
(ICCV 2025), included as a **git submodule and used unmodified**:

```
mimic/appearance_augmentation/third_party/Light-A-Video   ← upstream, pinned
```

Everything around it is ours. Upstream's entry points target short single-shot
demo clips — one hand-written prompt per YAML, a center-crop to a fixed
resolution, an 8 fps write — none of which suits navigation training data, where
cropping breaks the frame-to-action correspondence. Our layer adds:

- a **171-prompt pool** across 15 selectable categories, sampled per segment, so
  one clip sweeps several lighting conditions instead of one;
- **YOLOv8 person detection** with soft-mask blending, so pedestrians are not
  smeared or deformed by relighting (with cascade and full-frame fallbacks);
- **native resolution and fps preservation**, plus high-frequency detail carried
  over from the source;
- **arbitrary-length video** via segment/chunk scheduling and reassembly;
- **batch and multi-GPU drivers**, with per-video seed derivation for diversity.

```bash
uv sync --extra appearance   # torch, diffusers, ultralytics — needs a CUDA GPU

# 4 lighting conditions sampled from the pool, 8s of video each
uv run python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --output clip_aug.mp4 --n_lights 4

# restrict to night and rain conditions
uv run python -m mimic.appearance_augmentation.augment_video \
    --input clip.mp4 --categories night rain

# preview the sampled prompts without loading models (no GPU needed)
uv run python -m mimic.appearance_augmentation.augment_video --input clip.mp4 --dry_run
```

Weights download on first run into `mimic/appearance_augmentation/models/`.
See [`mimic/appearance_augmentation/README.md`](mimic/appearance_augmentation/README.md)
for the full upstream-vs-ours comparison, prompt categories, quality knobs and
the batching API.

### Action augmentation

Recorded data only shows the robot driving well, so a policy trained on it has
never seen itself off-course. This module manufactures that experience: the
robot drifts laterally off the recorded path and corrects back onto it, and both
the re-rendered video and the matching action label are generated.

```
offset
  ^
s |         ___
  |       /     \
  |     /         \
0 +----/-----------\------------->  t
  0        2 s      4 s
  deviate  peak     rejoined
```

Only the lateral offset is specified — heading is derived from the path tangent,
so the maneuver stays kinematically consistent. Peak yaw follows in closed form
as `atan(s·π / (H·v))`.

View synthesis is
[TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter)
(ICCV 2025), again a **submodule used unmodified**:

```
mimic/action_augmentation/third_party/TrajectoryCrafter   ← upstream, pinned
```

Upstream has no way to express a ground-plane maneuver — its pose generators are
orbit parameterizations for cinematic camera moves — and no notion of action
labels at all. Our layer adds the metric SE(2) maneuver, the derived heading, and
the label generation.

```bash
uv sync --extra action

# 0.5 m drift left, rejoining over a 4 s horizon
uv run python -m mimic.action_augmentation.augment_action \
    --input clip.mp4 --strength 0.5

# sample the offset per clip, coin-flip the side
uv run python -m mimic.action_augmentation.augment_action \
    --input clip.mp4 --strength_range 0.3 0.8 --seed 42

# trajectory and label only, no video synthesis (no GPU needed)
uv run python -m mimic.action_augmentation.augment_action \
    --input clip.mp4 --strength 0.5 --labels_only
```

Each clip reads an action-label sidecar (`.npy` / `.npz` / `.json`) beside the
video and writes an augmented one alongside the generated clip. The trajectory
math is pure numpy and covered by `tests/`; see
[`mimic/action_augmentation/README.md`](mimic/action_augmentation/README.md) for
the label schema, the two modes and the `--scale` calibration.

## Citation

If you find MIMIC helpful for your research, please cite:

```bibtex
@article{he2026learning,
  title={Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion},
  author={He, Honglin and Ma, Yukai and Squicciarini, Brad and Wu, Wayne and Zhou, Bolei},
  journal={arXiv preprint arXiv:2603.22527},
  year={2026}
}
```

## Acknowledgements

Appearance augmentation builds on **[Light-A-Video](https://github.com/bcmi/Light-A-Video)**
(Zhou, Bu, Ling et al., ICCV 2025), a training-free video relighting framework.
It is included as a git submodule and used **unmodified** — IC-Light relighting,
the Wan2.1 / AnimateDiff backbones, Consistent Light Attention and Progressive
Light Fusion are all theirs. The sidewalk-specific pipeline around it (prompt
pool, YOLO foreground preservation, resolution/fps preservation, long-video
scheduling, batch drivers) is ours. We thank the authors for releasing their
code under the Apache 2.0 license.

Action augmentation builds on **[TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter)**
(Yu, Hu, Xing & Shan, ICCV 2025), which redirects camera trajectory in
monocular video. It too is a submodule used **unmodified** — DepthCrafter depth
estimation, the point-cloud forward warp and the CogVideoX-Fun refinement are all
theirs. The ground-plane maneuver model, the derived-heading trajectory math and
the action-label generation are ours.

```bibtex
@InProceedings{Yu_2025_ICCV,
  title={TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models},
  author={Yu, Mark and Hu, Wenbo and Xing, Jinbo and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025},
  pages={100--111}
}
```

```bibtex
@InProceedings{Zhou_2025_ICCV,
  title={Light-A-Video: Training-free Video Relighting via Progressive Light Fusion},
  author={Zhou, Yujie and Bu, Jiazi and Ling, Pengyang and Zhang, Pan and Wu, Tong and Huang, Qidong and Li, Jinsong and Dong, Xiaoyi and Zang, Yuhang and Cao, Yuhang and Rao, Anyi and Wang, Jiaqi and Niu, Li},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025},
  pages={13315--13325}
}
```
