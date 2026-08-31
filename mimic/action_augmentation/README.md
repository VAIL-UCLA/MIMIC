# Action Augmentation

Corrective behavior expansion. Takes a recorded clip and its action label,
builds a **deviate-and-recover** maneuver on top of it, re-renders the video
from that camera path, and writes the matching action label.

The recorded data only shows the robot driving well. A policy trained on it has
never seen itself off-course and has no idea how to get back. This module
manufactures exactly that experience: the robot drifts off the recorded path and
corrects.

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

Only the **lateral** offset is specified. Heading is *derived* — yaw at each
sample is the tangent of the resulting path, so the maneuver is kinematically
consistent rather than the robot sliding sideways. Peak yaw follows in closed
form as `atan(s·π / (H·v))`; for a 0.5 m offset over a 4 s horizon at 1 m/s that
is 21.5°.

View synthesis is
[TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter)
(ICCV 2025), included as a **git submodule under `third_party/` and used
unmodified**.

## What is ours vs. upstream

Upstream supplies the view synthesis: DepthCrafter monocular depth, a
point-cloud forward warp to the target camera, and CogVideoX-Fun refinement of
the warped result. We changed none of it.

What upstream does *not* provide is any way to say "drive this ground-plane
maneuver". Its two pose generators are a single interpolated `target_pose` and a
spherical `theta/phi/r` keyframe file — orbit parameterizations aimed at
cinematic camera moves. Neither expresses a per-frame lateral offset with
derived heading, and neither knows anything about action labels.

| | Official TrajectoryCrafter | MIMIC |
|---|---|---|
| **Camera path** | one interpolated target pose, or spherical `theta/phi/r` keyframes | per-frame ground-plane SE(2) maneuver, injected via a bound `get_poses` |
| **Parameterization** | orbit angles around the scene | metric lateral offset in meters; heading derived from the path tangent |
| **Action labels** | none — it is a video model | label generated with the video and written as a sidecar |
| **Entry point** | `inference.py`, one clip per invocation | `augment_action.py`, with a `--labels_only` path that needs no GPU |
| **Randomization** | fixed seed and pose per run | offset magnitude and side sampled per clip from a range |

## Layout

| Path | Origin | Purpose |
|---|---|---|
| `third_party/TrajectoryCrafter` | upstream, unmodified | View synthesis (submodule, pinned) |
| `trajectory.py` | ours | The math: offset profiles, SE(2), tangent yaw, label generation |
| `labels.py` | ours | Sidecar I/O (`.npy` / `.npz` / `.json`) |
| `augment_action.py` | ours | CLI and API |
| `calibrate.py` | ours | Oracle-scale math: linear scale solve, affine-in-disparity fit |
| `calibrate_scale.py` | ours | Calibration CLI — video + labels in, oracle scale out |

## Setup

```bash
git submodule update --init --recursive
uv sync --extra action        # torch, diffusers, transformers — needs a CUDA GPU
```

The label path (`--labels_only`) is pure numpy and needs neither.

TrajectoryCrafter pulls ~52 GB of checkpoints from HuggingFace on first run:
CogVideoX-Fun-5b, DepthCrafter, BLIP-2 and Stable Video Diffusion.

> **Note:** upstream's own nested submodule (`DepthCrafter`) is declared with an
> SSH URL, so `--recursive` needs GitHub SSH access. Without it, run
> `git submodule update --init` (non-recursive) — the depth model is loaded from
> HuggingFace at runtime, not from that checkout.

## Label sidecars

Each clip carries a label file beside it. `.npy`, `.npz` and `.json` all work,
with two schemas:

| Field | Shape | Meaning |
|---|---|---|
| `poses` | `(N, 3)` | Ego poses `(x, y, yaw)` in a clip-global frame, one per frame. **Preferred.** |
| `waypoints` | `(N, K, 3)` | Per-frame future waypoints in each frame's ego frame — MIMIC's native label |
| `times` | `(N,)` | Optional timestamps in seconds; defaults to `--fps` (5 Hz) |

A bare `.npy` is read positionally: `(N, 3)` is poses, `(N, K, 3)` is waypoints.

Given only `waypoints`, a global path is reconstructed by chaining per-frame
steps. That is **approximate** — each step interpolates the label linearly, so on
a curved path it cuts the chord and the error accumulates (~7 cm over 10 s on a
10 m-radius arc at walking pace). The CLI says so when it happens. Supply
`poses` when you can.

Output is written as `.npz` with `poses`, `times`, `waypoints`, `label_times`
and a `metadata` JSON blob recording the strength, mode, profile and seed.

## Usage

```bash
# 0.5 m drift left, rejoining over a 4 s horizon
uv run python -m mimic.action_augmentation.augment_action \
    --input clip.mp4 --strength 0.5

# sample |offset| from [0.3, 0.8] m and coin-flip the side, per clip
uv run python -m mimic.action_augmentation.augment_action \
    --input clip.mp4 --strength_range 0.3 0.8 --seed 42

# trajectory and label only, no video synthesis (no GPU)
uv run python -m mimic.action_augmentation.augment_action \
    --input clip.mp4 --strength 0.5 --labels_only

# preview without writing anything
uv run python -m mimic.action_augmentation.augment_action \
    --input clip.mp4 --strength 0.5 --dry_run
```

Defaults are `<stem>_act.mp4` and `<stem>_act.npz` beside the input.

Like appearance augmentation, the seed is derived from `--seed` **and the input
path**, so a batch under one seed still varies per clip. `--fixed_seed` disables
that.

## Modes

| `--mode` | Behavior |
|---|---|
| `deviate_recover` (default) | Builds the S-curve maneuver and labels it. The video shows the robot drifting and correcting. |
| `reexpress` | Holds a constant lateral offset and re-expresses the *recorded* path in that frame. Pure recovery label, no maneuver of its own — the label simply points back to the lane. |

## Calibrating `--scale`

DepthCrafter produces relative, not metric, depth, so a lateral offset in meters
has to be converted into the reconstruction's units before it can be rendered.
`--scale` is that conversion, and guessing it silently mis-sizes every generated
maneuver.

Do not guess it — the action labels already record how far the robot actually
moved, so the scale can be read off ground truth:

```bash
# depth computed with DepthCrafter, exactly as the renderer would
uv run python -m mimic.action_augmentation.calibrate_scale --input clip.mp4

# reuse a precomputed depth stack (N, H, W) — no GPU needed
uv run python -m mimic.action_augmentation.calibrate_scale \
    --input clip.mp4 --depth depth.npy

# calibrate across a corpus and save the report
uv run python -m mimic.action_augmentation.calibrate_scale \
    --input 'data/*/front_*.mp4' --depth_dir depths/ --out scale.json
```

It prints the number to pass straight back as `--scale`. The scale is a property
of the camera and the depth model, not of the clip, so calibrate once over a
handful of clips and keep it fixed.

### How it works

For a static scene point at pixel `p` in frame A with relative depth `Z_rel`, its
metric position in camera A is `(Z_rel / s) · ray`. Applying the label-derived
rigid motion `(R, t)` into frame B:

```
X_B = (R · Z_rel · ray + s · t) / s
```

Projection ignores the positive factor `1/s`, so with `Y = R · Z_rel · ray` the
observed pixel `p'` must satisfy

```
p'_h  ×  (K Y + s K t)  =  0
```

which is **linear in `s`**. Every tracked feature contributes one such
constraint and the least-squares solution over all of them is the scale. There
is no small-angle approximation — the label's yaw is applied exactly — and
frames where the robot barely moves are skipped, since `t → 0` leaves `s`
unconstrained.

Features are tracked with Lucas-Kanade plus a forward-backward consistency
check, worst-residual points are trimmed, and per-pair estimates are combined
with a median.

### Reading the output

```
    pairs used     : 39 (0 skipped, low baseline)
    scale          : 2.5012 depth units / meter
    spread (MAD)   : 0.5%  [p16 2.483 .. p84 2.519]
    reproj error   : 0.097 px
    depth range    : 97.2 .. 137.2 (median 112.9)
    affine fit     : alpha 0.3994, beta +0.00000
    scale drift    : 0.0% across that range (2.504 near .. 2.504 far)
      -> a single --scale fits this clip.
```

| Line | What it tells you |
|---|---|
| `spread (MAD)` | How much the frame pairs disagree. Above ~20% means something is wrong — most likely the labels are not metric, or not time-aligned with the video |
| `reproj error` | Fit quality in pixels. Sub-pixel is healthy |
| `scale drift` | Whether one number suffices — see below |

### When one scale is not enough

DepthCrafter is affine-invariant in *disparity*, and upstream's post-processing
(normalize across the clip, then invert) preserves that:

```
1 / Z_rel  =  alpha' / Z_metric  +  beta
```

A single multiplicative scale is exact only when `beta = 0`. Otherwise the
meters-per-unit conversion varies with depth, and any one `--scale` is a local
linearization — right mid-scene, off at the extremes. The calibrator fits both
parameters and reports how far the conversion drifts across the depth range the
clip actually covers. Under ~25% drift, one number is fine; above it, expect
offsets to be mis-sized for near and far content.

The headline `scale` remains the least-squares single value, because that is the
model the renderer implements. The affine fit is a trust diagnostic, not a
second knob.

> Note: `--scale` affects the **rendered video only**. The labels are metric and
> exact regardless of it — they come from `trajectory.py`, which never touches
> the depth reconstruction.

## Knobs

| Flag | Default | Effect |
|---|---|---|
| `--strength` | — | Peak lateral offset in meters; positive left, negative right |
| `--strength_range LO HI` | — | Sample `|offset|` from the range, side coin-flipped |
| `--horizon` | 4.0 | Maneuver duration in seconds; peak at half |
| `--start_time` | 0.0 | When the maneuver begins, in clip time |
| `--profile` | `raised_cosine` | `raised_cosine` (smooth, zero lateral velocity at both ends and the peak), `smoothstep`, or `triangle` (velocity steps at the peak — baseline only) |
| `--scale` | 1.0 | Depth units per meter — see above |
| `--diffusion_inference_steps` | 50 | Refinement steps |

## Testing

The math is pure numpy and covered by `tests/test_trajectory.py`:

```bash
uv run --extra dev pytest tests/ -q
```

The tests pin the properties that matter — the offset returns to zero, the peak
equals the requested strength, the displacement is perpendicular on curved paths,
yaw matches the closed-form `atan(s·π/(H·v))`, labels are ego-frame invariant,
and the waypoint round-trip holds within its documented drift.
