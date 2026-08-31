# Sample clips

Three short sidewalk clips from the Coco Robotics delivery-robot fleet, kept in
the repo as fixtures for the augmentation and calibration tooling. Each folder
is named by a random UID; the original scenario identifiers are not published.

```
assets/clips/<uid>/
├── meta.json           camera intrinsics, fps, frame count, provenance
├── poses_recorded.npy  (N, 5) float32 — x_m, y_m, yaw_rad, v_mps, w_radps
├── rgb_pinhole.mp4     rectified pinhole stream (the one to work from)
├── rgb_fisheye.mp4     raw Kannala-Brandt fisheye stream
├── route.mp4           synthetic route/map render
└── semantics.zpack     zstd-compressed zip of a (N, H, W) uint8 class map
```

All streams are 480 × 270 at 20 fps, roughly 19 s each. `poses_recorded.npy`
is in a world frame with z up, so it is directly usable as the action-label
sidecar for the calibration script:

```bash
uv run python -m mimic.action_augmentation.calibrate_scale \
    --input assets/clips/<uid>/rgb_pinhole.mp4 \
    --labels assets/clips/<uid>/poses_recorded.npy \
    --fps 20
```

## Anonymization

People in the two RGB streams have had their head regions blurred by
[`scripts/anonymize_clips.py`](../../scripts/anonymize_clips.py). Each clip's
`meta.json` carries an `anonymization` block recording the detector settings and
the measured coverage.

Note that a plain face detector is not sufficient at this resolution — at
480 × 270 and at night, faces are only a handful of pixels and CenterFace (the
detector behind `deface`) fires on almost none of them. People are therefore
located as *people*, by a YOLOv8 person detector and by the `ACTOR_PERSON`
channel of the shipped semantic masks, and the head end of each detection is
blurred. Face detection runs on top of that as a supplement. See the script's
module docstring for the measurements behind that choice.

`route.mp4` is a vector map render with no camera imagery and is left as-is.
