"""Materialize the window of a clip that the renderer will actually see.

TrajectoryCrafter reads the *leading* ``TC_VIDEO_LENGTH * stride`` frames of
whatever video it is handed. Anything later in the recording is invisible to
it. That is a problem for real footage: a robot that idles at a crossing for
the first eight seconds has its only usable maneuver outside the window, and no
choice of ``--start_time`` can bring it in.

Widening the window with ``--stride`` trades away the thing being measured —
past about 5 effective fps the frame-to-frame motion is large enough that
feature tracking stops finding correspondences, so the depth scale can no
longer be calibrated at all.

So instead of moving the window, move the clip: write out the frames the window
covers as a small self-contained bundle — video, poses, and a ``meta.json``
carrying the true frame rate — and point the pipeline at that. Every downstream
step then sees a clip whose leading window *is* the interesting stretch, with
no special cases anywhere else.

The bundle is a normal clip folder, so sidecar discovery, frame-rate discovery
and scale calibration all work on it unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import labels as label_io

#: Names inside a materialized bundle. ``poses_recorded.npy`` is one of the
#: names :data:`~.labels.SIDECAR_NAMES` already looks for.
VIDEO_NAME = "clip.mp4"
POSES_NAME = "poses_recorded.npy"
META_NAME = "meta.json"


def frame_span(length: int, stride: int) -> int:
    """How many source frames a window covers.

    The renderer samples ``length`` frames at ``stride``, so it touches
    ``(length - 1) * stride + 1`` frames, not ``length * stride``.
    """
    return (length - 1) * max(stride, 1) + 1


def window_start_for(
    maneuver_start_s: float,
    horizon_s: float,
    fps: float,
    length: int,
    stride: int,
    n_frames: int,
) -> int:
    """First frame of a window that contains the whole maneuver.

    Returns 0 when the maneuver already fits in the leading window. Otherwise
    the window is placed to start on the maneuver, then pulled back if that
    would run past the end of the recording.
    """
    span = frame_span(length, stride)
    end_frame = int(np.ceil((maneuver_start_s + horizon_s) * fps))
    if end_frame <= span:
        return 0
    start = int(np.floor(maneuver_start_s * fps))
    return int(max(0, min(start, n_frames - span)))


def _read_frames(video_path: Path, start: int, count: int) -> tuple[list, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Seeking by frame index is unreliable on some containers, and these
        # clips are short, so read forward and keep the slice.
        frames, index = [], 0
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= start:
                frames.append(frame)
            index += 1
    finally:
        cap.release()
    return frames, float(fps)


def _write_video(path: Path, frames: list, fps: float) -> None:
    """Write frames losslessly enough that the re-encode is not visible.

    These frames are the renderer's only view of the scene, so a default-quality
    re-encode would put compression artifacts into the thing being augmented.
    Prefers x264 at near-lossless quality; falls back to OpenCV's writer.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None

    if imageio is not None:
        try:
            writer = imageio.get_writer(
                str(path), fps=fps, codec="libx264", quality=None,
                macro_block_size=1, pixelformat="yuv420p", ffmpeg_params=["-crf", "10"],
            )
            try:
                for frame in frames:
                    writer.append_data(frame[:, :, ::-1])  # BGR from cv2 -> RGB
            finally:
                writer.close()
            return
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the run
            print(f"note: ffmpeg writer unavailable ({exc}); using OpenCV", flush=True)

    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def materialize(
    video_path: Path,
    dest: Path,
    start: int,
    length: int,
    stride: int,
    fps: float | None = None,
    sidecar: Path | None = None,
) -> Path:
    """Write the window ``[start, start + span)`` of a clip as its own bundle.

    Returns the path of the bundle's video. The poses are sliced to match and
    re-based so the bundle's timeline starts at zero.
    """
    video_path = Path(video_path)
    dest = Path(dest)
    span = frame_span(length, stride)

    rate = fps or label_io.clip_fps(video_path) or label_io.DEFAULT_FPS

    frames, src_fps = _read_frames(video_path, start, span)
    if not frames:
        raise ValueError(f"{video_path.name}: no frames at offset {start}")

    dest.mkdir(parents=True, exist_ok=True)
    out_video = dest / VIDEO_NAME
    _write_video(out_video, frames, src_fps or rate)

    sidecar = sidecar or label_io.find_sidecar(video_path)
    data = label_io.load_labels(sidecar, fps=rate)
    stop = min(start + len(frames), len(data.poses))
    poses = np.asarray(data.poses[start:stop], dtype=np.float64)
    if len(poses) == 0:
        raise ValueError(f"{sidecar.name}: no poses at offset {start}")
    np.save(dest / POSES_NAME, poses)

    (dest / META_NAME).write_text(json.dumps({
        "fps": rate,
        "window": {"start": start, "length": length, "stride": stride},
        "source_frames": [start, stop],
        "note": "Materialized window of a longer recording; times are re-based to 0.",
    }, indent=2))
    return out_video
