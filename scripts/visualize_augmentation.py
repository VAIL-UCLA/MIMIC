#!/usr/bin/env python
"""Side-by-side view of what the two augmentation stages did to a clip.

Five panels, left to right:

    RGB            the recorded frame
    trajectory     the recorded label, bird's-eye, from that frame
    appearance     the same frame relit by stage 1
    action         the same frame re-rendered by stage 2
    new trajectory the augmented label, with the recorded one behind it

Writes an mp4 per clip, and a contact sheet if asked. Panels whose stage has not
been run are drawn as a labelled placeholder rather than skipped, so the layout
stays the same and it is obvious what is missing.

    uv run python scripts/visualize_augmentation.py \\
        --input assets/clips --action_dir out/action --output out/viz --fps 20

Only needs opencv and numpy; no GPU, no models.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stage_common import (
    clip_label,
    find_clips,
    human_time,
    progress,
    say,
    step,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from mimic.action_augmentation import labels as label_io
from mimic.action_augmentation import trajectory as tj

STAGE = "Visualize"

#: Height every panel is drawn at. RGB panels keep their aspect ratio; the
#: bird's-eye panels are square.
PANEL_H = 288

#: Bird's-eye extent: meters ahead of the robot, and meters to each side.
BEV_FORWARD_M = 10.0
BEV_LATERAL_M = 5.0

#: Height of the caption strip above the panels.
CAPTION_H = 26

BG = (24, 24, 28)
GRID = (58, 58, 66)
AXIS = (96, 96, 108)
TEXT = (232, 232, 236)
MUTED = (128, 128, 138)
ORIGINAL = (150, 190, 90)      # BGR — recorded path
AUGMENTED = (90, 140, 250)     # BGR — augmented path
ROBOT = (240, 240, 240)


def read_all_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def fit_panel(frame: np.ndarray, height: int = PANEL_H) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (round(w * height / h), height),
                      interpolation=cv2.INTER_AREA)


def placeholder(width: int, height: int, lines: list[str]) -> np.ndarray:
    """A panel for a stage that has not been run."""
    panel = np.full((height, width, 3), BG, np.uint8)
    cv2.rectangle(panel, (6, 6), (width - 7, height - 7), GRID, 1, cv2.LINE_AA)
    y = height // 2 - (len(lines) - 1) * 11
    for i, line in enumerate(lines):
        colour = MUTED if i else TEXT
        scale = 0.42 if i else 0.5
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(panel, line, ((width - tw) // 2, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    return panel


def bev_canvas() -> tuple[np.ndarray, float, int, int]:
    """Empty bird's-eye canvas, with the robot at the bottom centre facing up.

    Returns ``(canvas, pixels_per_meter, origin_x, origin_y)``.
    """
    size = PANEL_H
    ppm = min(size / BEV_FORWARD_M, size / (2.0 * BEV_LATERAL_M))
    ox, oy = size // 2, size - 18
    canvas = np.full((size, size, 3), BG, np.uint8)

    for metres in range(2, int(BEV_FORWARD_M) + 1, 2):
        y = round(oy - metres * ppm)
        if y < 0:
            break
        cv2.line(canvas, (0, y), (size, y), GRID, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{metres}m", (4, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, MUTED, 1, cv2.LINE_AA)
    for metres in (-4, -2, 2, 4):
        x = round(ox + metres * ppm)
        if 0 <= x < size:
            cv2.line(canvas, (x, 0), (x, oy), GRID, 1, cv2.LINE_AA)
    cv2.line(canvas, (ox, 0), (ox, oy), AXIS, 1, cv2.LINE_AA)
    return canvas, ppm, ox, oy


def to_pixels(points: np.ndarray, ppm: float, ox: int, oy: int) -> np.ndarray:
    """Ego-frame ``(x forward, y left)`` metres to canvas pixels."""
    xs = ox - points[:, 1] * ppm      # +y is left, and left is -x on screen
    ys = oy - points[:, 0] * ppm      # +x is forward, and forward is up
    return np.stack([xs, ys], axis=-1).astype(np.int32)


def draw_track(canvas, points, ppm, ox, oy, colour, thickness=2, dots=True):
    if len(points) == 0:
        return
    pts = to_pixels(points, ppm, ox, oy)
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, colour, thickness, cv2.LINE_AA)
    if dots:
        for x, y in pts:
            cv2.circle(canvas, (int(x), int(y)), 2, colour, -1, cv2.LINE_AA)


def draw_robot(canvas, ox, oy):
    cv2.drawMarker(canvas, (ox, oy), ROBOT, cv2.MARKER_TRIANGLE_UP, 11, 2, cv2.LINE_AA)


def bev_panel(waypoints, reference=None, note: str = "") -> np.ndarray:
    """One bird's-eye panel: a track, optionally over a faint reference track."""
    canvas, ppm, ox, oy = bev_canvas()
    if reference is not None:
        draw_track(canvas, reference[:, :2], ppm, ox, oy, ORIGINAL, 1, dots=False)
    draw_track(canvas, waypoints[:, :2], ppm, ox, oy,
               AUGMENTED if reference is not None else ORIGINAL)
    draw_robot(canvas, ox, oy)
    if reference is not None:
        cv2.putText(canvas, "recorded", (PANEL_H - 74, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, ORIGINAL, 1, cv2.LINE_AA)
        cv2.putText(canvas, "augmented", (PANEL_H - 79, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, AUGMENTED, 1, cv2.LINE_AA)
    if note:
        cv2.putText(canvas, note, (6, PANEL_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT, 1, cv2.LINE_AA)
    return canvas


def future_in_frame(poses, times, label_times, index, reference) -> np.ndarray:
    """Sample a path at ``t[index] + label_times``, in ``reference``'s frame.

    The stored labels each sit in their *own* ego frame, and the augmented frame
    differs from the recorded one by up to the maneuver's yaw. Drawing the two
    on one canvas therefore has to re-express them against a common origin, or a
    15 degree heading difference over a 12 m lookahead shows up as metres of
    apparent deviation that the robot never travelled.
    """
    t = times[index] + np.asarray(label_times, dtype=np.float64)
    x = np.interp(t, times, poses[:, 0])
    y = np.interp(t, times, poses[:, 1])
    yaw = np.interp(t, times, np.unwrap(poses[:, 2]))
    c, sn = np.cos(reference[2]), np.sin(reference[2])
    dx, dy = x - reference[0], y - reference[1]
    return np.stack([c * dx + sn * dy, -sn * dx + c * dy,
                     tj.wrap_angle(yaw - reference[2])], axis=-1)


def lateral_offset_at(recorded, augmented, index) -> float:
    """How far off course the augmented path is right now, in meters.

    Taken between the two paths in the world, projected on the recorded
    heading — this is the maneuver's own offset, not an artifact of the frames.
    """
    ref = recorded[index]
    dx, dy = augmented[index, 0] - ref[0], augmented[index, 1] - ref[1]
    return float(-np.sin(ref[2]) * dx + np.cos(ref[2]) * dy)


def caption_strip(width: int, title: str, highlight: bool = False) -> np.ndarray:
    strip = np.full((CAPTION_H, width, 3), BG, np.uint8)
    cv2.putText(strip, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                TEXT if highlight else MUTED, 1, cv2.LINE_AA)
    return strip


def compose(panels: list[tuple[str, np.ndarray, bool]], header: str) -> np.ndarray:
    """Stack captions over panels and lay them out in one row."""
    columns = []
    for title, panel, highlight in panels:
        columns.append(np.vstack([caption_strip(panel.shape[1], title, highlight), panel]))
    row = np.hstack(columns)
    banner = np.full((CAPTION_H, row.shape[1], 3), BG, np.uint8)
    cv2.putText(banner, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                TEXT, 1, cv2.LINE_AA)
    return np.vstack([banner, row])


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


#: Upstream writes the refined clip under this name inside its output folder.
RENDER_NAME = "gen.mp4"


def matching_output(
    directory: Path | None, clip: Path, suffix: str, ext: str = ".mp4"
) -> Path | None:
    """Find a stage's output for this clip, whatever variant naming it used.

    Labels and video are looked up separately: ``--labels_only`` runs of stage 2
    produce a sidecar and no video, and the trajectory panel is drawn from the
    sidecar alone.
    """
    if directory is None:
        return None
    stem = clip.parent.name if clip.stem.startswith("rgb_") else clip.stem
    exact = directory / f"{stem}{suffix}{ext}"
    if exact.is_file():
        return exact

    # A rendered clip is not a file at that path: upstream treats the stem as a
    # folder and writes gen.mp4 inside it, next to its intermediates.
    if ext == ".mp4":
        for folder in [directory / f"{stem}{suffix}",
                       *sorted(directory.glob(f"{stem}{suffix}*"))]:
            candidate = folder / RENDER_NAME
            if candidate.is_file():
                return candidate

    matches = sorted(directory.glob(f"{stem}{suffix}*{ext}"))
    return matches[0] if matches else None


def labelled_source(label_path: Path | None, clip: Path) -> Path:
    """The clip an augmented label actually describes.

    Stage 2 brings a maneuver into the renderer's window by materializing that
    stretch as its own bundle, so the label covers the window and not the whole
    recording. Comparing it against the full clip would line frame 0 of one up
    with frame 0 of the other, which are seconds apart — and index off the end.
    The label records the video it was made from, so follow it.
    """
    if label_path is None:
        return clip
    try:
        with np.load(label_path, allow_pickle=True) as data:
            if "metadata" not in data:
                return clip
            meta = json.loads(str(data["metadata"]))
    except (OSError, ValueError, KeyError):
        return clip
    source = meta.get("source_video")
    if not source:
        return clip
    candidate = Path(source)
    return candidate if candidate.is_file() else clip


def window_offset(source: Path, clip: Path) -> int:
    """Frame of ``clip`` that ``source``'s first frame corresponds to.

    Stage 1 runs on the whole recording, but stage 2 may have windowed it. Both
    outputs are then indexed by frame number, and those numbers mean different
    moments unless the window's offset is added back — the relit panel would
    otherwise show a point seconds away from the rest of the row.
    """
    if source == clip:
        return 0
    meta = source.parent / "meta.json"
    if not meta.is_file():
        return 0
    try:
        blob = json.loads(meta.read_text())
    except (OSError, ValueError):
        return 0
    span = blob.get("source_frames")
    if isinstance(span, list) and span and isinstance(span[0], int):
        return span[0]
    window = blob.get("window")
    if isinstance(window, dict) and isinstance(window.get("start"), int):
        return window["start"]
    return 0


def visualize_clip(clip: Path, args) -> dict:
    name = clip_label(clip)
    record = {"clip": name}

    label_path = matching_output(
        Path(args.action_dir) if args.action_dir else None, clip, "_act", ".npz"
    )
    source = labelled_source(label_path, clip)
    if source != clip:
        record["windowed_source"] = str(source)

    frames = read_all_frames(source)
    if not frames:
        raise ValueError("no frames decoded")

    sidecar = label_io.find_sidecar(source)
    fps = args.fps or label_io.clip_fps(source) or label_io.DEFAULT_FPS
    original = label_io.load_labels(sidecar, fps=fps)

    appearance_path = matching_output(
        Path(args.appearance_dir) if args.appearance_dir else None, clip, "_light"
    ) or matching_output(
        Path(args.appearance_dir) if args.appearance_dir else None, clip, ""
    )
    action_path = matching_output(
        Path(args.action_dir) if args.action_dir else None, clip, "_act"
    )
    record["appearance"] = str(appearance_path) if appearance_path else None
    record["action"] = str(action_path) if action_path else None

    # Stage 1 runs on the full recording; the action label may describe only a
    # window of it. Shift into the full clip's numbering for the relit panel.
    frame_offset = window_offset(source, clip)
    appearance_frames = read_all_frames(appearance_path) if appearance_path else []
    action_frames = read_all_frames(action_path) if action_path else []

    augmented = None
    if label_path is not None:
        augmented = label_io.load_labels(label_path, fps=fps)
    record["augmented_labels"] = str(label_path) if label_path else None

    first = max(0, min(args.start_frame, len(frames) - 1))
    last = len(frames) if args.frames is None else min(first + args.frames, len(frames))
    step_n = max(1, args.frame_stride)
    indices = list(range(first, last, step_n))
    if not indices:
        raise ValueError(f"no frames selected from {len(frames)} (start {first})")

    rgb_w = fit_panel(frames[0]).shape[1]
    speed = tj.path_speed(original.poses, original.times)

    composed = []
    for i in indices:
        rgb = fit_panel(frames[i])
        original_wp = original.waypoints[i] if original.waypoints is not None else None
        if original_wp is None:
            original_wp = tj.waypoints_from_path(
                original.poses, original.times, original.label_times
            )[i]

        panels = [("RGB", rgb, True),
                  ("trajectory (recorded)",
                   bev_panel(original_wp, note=f"{speed[i]:.1f} m/s"), False)]

        if appearance_frames:
            j = min(i + frame_offset, len(appearance_frames) - 1)
            panels.append(("appearance augmented", fit_panel(appearance_frames[j]), True))
        else:
            panels.append(("appearance augmented",
                           placeholder(rgb_w, PANEL_H,
                                       ["not generated", "run stage 1"]), False))

        if action_frames:
            j = min(i, len(action_frames) - 1)
            panels.append(("action augmented", fit_panel(action_frames[j]), True))
        else:
            panels.append(("action augmented",
                           placeholder(rgb_w, PANEL_H,
                                       ["not generated", "run stage 2 (needs a GPU)"]), False))

        if augmented is not None:
            # Both futures against the recorded pose, so the gap between them is
            # the deviation and nothing else.
            ref = original.poses[i]
            aug_future = future_in_frame(
                augmented.poses, augmented.times, original.label_times, i, ref
            )
            rec_future = future_in_frame(
                original.poses, original.times, original.label_times, i, ref
            )
            deviation = lateral_offset_at(original.poses, augmented.poses, i)
            panels.append(("trajectory (augmented)",
                           bev_panel(aug_future, reference=rec_future,
                                     note=f"{deviation:+.2f} m off course"), False))
        else:
            panels.append(("trajectory (augmented)",
                           placeholder(PANEL_H, PANEL_H,
                                       ["no augmented label", "run stage 2"]), False))

        header = f"{name}   frame {i}/{len(frames) - 1}   t={i / fps:5.2f}s"
        composed.append(compose(panels, header))

    out_dir = Path(args.output).expanduser().resolve()
    stem = clip.parent.name if clip.stem.startswith("rgb_") else clip.stem
    video_path = out_dir / f"{stem}_compare.mp4"
    write_video(video_path, composed, fps / step_n)
    record["output"] = str(video_path)

    if args.contact_sheet:
        picks = np.linspace(0, len(composed) - 1,
                            min(args.sheet_rows, len(composed))).astype(int)
        sheet = np.vstack([composed[k] for k in picks])
        sheet_path = out_dir / f"{stem}_compare.png"
        cv2.imwrite(str(sheet_path), sheet)
        record["contact_sheet"] = str(sheet_path)

    record["status"] = "written"
    record["frames"] = len(composed)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare a clip against its appearance- and action-augmented versions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Clip directory, video file, or glob (quote it).")
    parser.add_argument("--output", default="out/viz", help="Directory for the comparisons.")
    parser.add_argument("--stream", default="rgb_pinhole.mp4")
    parser.add_argument("--appearance_dir", default=None,
                        help="Stage 1 output directory. Omitted panels are placeholders.")
    parser.add_argument("--action_dir", default=None,
                        help="Stage 2 output directory.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate of the clip and its labels.")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="First source frame to compose. Point it at the "
                             "maneuver window to see the deviation.")
    parser.add_argument("--frames", type=int, default=None,
                        help="Compose this many source frames from --start_frame.")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Use every Nth frame, for a shorter, smaller video.")
    parser.add_argument("--contact_sheet", action="store_true",
                        help="Also write a PNG of evenly spaced frames stacked.")
    parser.add_argument("--sheet_rows", type=int, default=4,
                        help="Rows in the contact sheet.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    clips = find_clips(args.input, args.stream)
    if not clips:
        print(f"error: no clips matched {args.input!r}", file=sys.stderr)
        return 1

    print(f"\n\033[1m{STAGE} · augmentation comparison\033[0m")
    print(f"  {len(clips)} clip(s) -> {Path(args.output).resolve()}\n")

    records = []
    bar = progress(clips, "Composing", unit="clip")
    for clip in bar:
        with step(bar, f"Composing {clip_label(clip)}"):
            item_started = time.perf_counter()
            try:
                record = visualize_clip(clip, args)
            except Exception as exc:  # noqa: BLE001 - one bad clip must not sink the rest
                record = {"clip": clip_label(clip), "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}"}
            record["seconds"] = round(time.perf_counter() - item_started, 1)

        if record["status"] == "written":
            missing = [n for n, k in (("appearance", "appearance"),
                                      ("action video", "action"),
                                      ("augmented label", "augmented_labels"))
                       if not record.get(k)]
            note = f"  ({', '.join(missing)} missing)" if missing else ""
            say(bar, f"  ✓ {Path(record['output']).name}  {record['frames']} frames"
                     f"  [{human_time(record['seconds'])}]{note}")
        else:
            say(bar, f"  ✗ {record['clip']}  {record['error']}")
        records.append(record)
    bar.close()

    return summarize(STAGE, records, time.perf_counter() - started, None)


if __name__ == "__main__":
    raise SystemExit(main())
