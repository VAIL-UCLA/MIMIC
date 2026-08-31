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

People in the two RGB streams have had their head regions blurred. Each clip's
`meta.json` carries an `anonymization` block recording the exact detector
settings and the measured coverage.

A plain face detector is not sufficient at this resolution — at 480 × 270 and at
night, faces are only a handful of pixels, and CenterFace (the detector behind
`deface`) covered 1 of 276 person-frames at its default threshold on the clip
with the most pedestrian content. People were therefore located as *people*, by
a YOLOv8 person detector and by the `ACTOR_PERSON` channel of the shipped
semantic masks, and the head end of each detection blurred, with face detection
on top as a supplement. Measured against those semantic masks, head-pixel recall
was 100% on all three pinhole streams.

`route.mp4` is a vector map render with no camera imagery and was left as-is.

## Known quirks

These are recorded clips, not clean test fixtures, and the tooling is built to
cope with what is actually in them:

- **The robot parks.** It idles at crossings for seconds at a time — 40% of
  `09294dbb` and 37% of `38aee4d8` are stationary, including the first 3.9 s of
  `09294dbb`. A maneuver placed there would label a parked robot sliding
  sideways, so stage 2 picks a moving window instead.
- **The robot reverses**, at up to 2.2 m/s. A path tangent points backwards
  then, so headings are taken from the recorded yaw rather than the tangent.
- **A few headings are corrupt.** `298696ea` and `38aee4d8` each carry two
  samples whose recorded yaw is 180° from their direction of travel, at a wrap
  boundary. Stage 2 detects these and routes the maneuver around them.
