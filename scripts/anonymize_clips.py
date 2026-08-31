#!/usr/bin/env python
"""Blur identifiable people out of the sample clips in ``assets/clips``.

Run it with an ephemeral dependency overlay -- none of this is needed to use
the ``mimic`` package itself:

    uv run --with ultralytics --with deface --with opencv-python \
        python scripts/anonymize_clips.py assets/clips

Why not plain ``deface``
------------------------
``deface`` detects *faces* with CenterFace.  On this footage -- 480x270,
at night, pedestrians mostly 20-90 px tall -- it barely fires: measured
against the ``ACTOR_PERSON`` channel of ``semantics.zpack`` on the clip with
the most pedestrian content, CenterFace at its default threshold covered
1 of 276 person-frames, and even at a threshold low enough to fire on every
frame the great majority of its boxes were streetlights and signage rather
than people.  Faces here are simply below its resolution floor.

So people are found as *people* instead, and the head end of each one is
blurred:

  1. YOLOv8 ``person`` boxes, on both RGB streams.
  2. The ``ACTOR_PERSON`` class of the shipped semantic masks -- ground truth,
     but defined on the pinhole stream only.
  3. CenterFace boxes on top, to catch a face the person detectors miss
     (a face on a poster, a driver through a windscreen).

The union is padded by a few frames in time to bridge single-frame dropouts,
mosaicked and blurred, and composited back through a feathered mask.
``route.mp4`` is a synthetic map render with no camera imagery, so it is left
alone rather than re-encoded for nothing.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

#: Class id of people in the shipped semantic taxonomy.
ACTOR_PERSON = 2

#: Streams to anonymize. ``route.mp4`` is a map render and is skipped.
RGB_STREAMS = ("rgb_pinhole.mp4", "rgb_fisheye.mp4")

#: Which stream the semantic masks are defined on.
SEMANTIC_SOURCE = "rgb_pinhole.mp4"

#: A person box taller than this gets only its head end blurred; anything
#: smaller is blurred whole, since the head is then a couple of pixels and
#: locating it inside the box is meaningless.
WHOLE_BODY_BELOW_PX = 24

#: Fraction of a person box height treated as the head end.
HEAD_FRACTION = 0.40

#: Grow every region by this fraction of its size before blurring.
REGION_MARGIN = 0.25

#: Union each frame's regions with those of its neighbours this many frames
#: either side, so a one-frame detector dropout still stays covered.
TEMPORAL_PAD = 3

#: Mosaic cell size, as a divisor of frame width. 12 turns a 480 px-wide
#: frame into 40 px-wide blocks -- a 12 px face lands in a single cell.
MOSAIC_DIVISOR = 12

#: Feather radius of the composite mask, in pixels.
FEATHER_PX = 3

#: Weight cache shared with the appearance module (gitignored).
DEFAULT_YOLO_WEIGHTS = (
    Path(__file__).resolve().parent.parent
    / "mimic" / "appearance_augmentation" / "models" / "yolov8x.pt"
)


def load_semantic_person(clip: Path) -> np.ndarray | None:
    """Boolean (frames, H, W) person mask from ``semantics.zpack``, if present.

    The file is a zstd-compressed zip holding a single ``data.npy``.
    """
    path = clip / "semantics.zpack"
    if not path.is_file():
        return None
    try:
        import zstandard
    except ImportError:
        raw = subprocess.run(
            ["zstd", "-q", "-dc", str(path)], check=True, capture_output=True
        ).stdout
    else:
        raw = zstandard.ZstdDecompressor().stream_reader(path.open("rb")).read()
    with zipfile.ZipFile(io.BytesIO(raw)).open("data.npy") as fh:
        sem = np.load(io.BytesIO(fh.read()))
    return sem == ACTOR_PERSON


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, fps


def write_video(path: Path, frames: list[np.ndarray], fps: float, crf: int) -> None:
    """Encode with ffmpeg over a pipe, so the codec settings are explicit."""
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def head_region(box, shape) -> tuple[int, int, int, int] | None:
    """Head end of a person box, clipped to the frame and given a margin."""
    h_img, w_img = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    height = y2 - y1
    if height > WHOLE_BODY_BELOW_PX:
        y2 = y1 + height * HEAD_FRACTION
    mx = (x2 - x1) * REGION_MARGIN
    my = (y2 - y1) * REGION_MARGIN
    x1, x2 = x1 - mx, x2 + mx
    y1, y2 = y1 - my, y2 + my
    xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    xi2, yi2 = min(w_img, int(np.ceil(x2))), min(h_img, int(np.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return xi1, yi1, xi2, yi2


def mask_person_pixels(person: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the connected person blobs in one semantic frame."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(person.astype(np.uint8), 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        boxes.append((x, y, x + w, y + h))
    return boxes


def detect(frames, semantic_person, yolo_weights, yolo_conf, face_thresh, imgsz):
    """Per-frame head/face regions from all three sources, plus a source tally."""
    from deface.centerface import CenterFace
    from ultralytics import YOLO

    shape = frames[0].shape
    model = YOLO(yolo_weights)
    centerface = CenterFace(in_shape=(shape[1] * 2, shape[0] * 2))

    regions: list[list[tuple[int, int, int, int]]] = [[] for _ in frames]
    counts = {"yolo": 0, "semantic": 0, "face": 0}

    for i, frame in enumerate(frames):
        result = model.predict(
            frame, classes=[0], conf=yolo_conf, imgsz=imgsz, verbose=False
        )[0]
        for box in result.boxes.xyxy.cpu().numpy():
            region = head_region(box, shape)
            if region:
                regions[i].append(region)
                counts["yolo"] += 1

        if semantic_person is not None and i < len(semantic_person):
            for box in mask_person_pixels(semantic_person[i]):
                region = head_region(box, shape)
                if region:
                    regions[i].append(region)
                    counts["semantic"] += 1

        dets, _ = centerface(frame, threshold=face_thresh)
        for det in dets:
            # A face box is already the head -- take it whole, with margin.
            x1, y1, x2, y2 = det[:4]
            mx, my = (x2 - x1) * REGION_MARGIN, (y2 - y1) * REGION_MARGIN
            xi1, yi1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            xi2 = min(shape[1], int(np.ceil(x2 + mx)))
            yi2 = min(shape[0], int(np.ceil(y2 + my)))
            if xi2 > xi1 and yi2 > yi1:
                regions[i].append((xi1, yi1, xi2, yi2))
                counts["face"] += 1

    return regions, counts


def pad_in_time(regions, pad):
    """Union each frame's regions with its temporal neighbours'."""
    n = len(regions)
    return [
        [r for j in range(max(0, i - pad), min(n, i + pad + 1)) for r in regions[j]]
        for i in range(n)
    ]


def blur_frame(frame, regions):
    """Composite a mosaicked, blurred copy through a feathered elliptical mask."""
    if not regions:
        return frame, np.zeros(frame.shape[:2], bool)

    h, w = frame.shape[:2]
    cell_w = max(1, w // MOSAIC_DIVISOR)
    cell_h = max(1, h // MOSAIC_DIVISOR)
    small = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    obscured = cv2.GaussianBlur(mosaic, (0, 0), max(2.0, w / 120.0))

    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in regions:
        centre = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, -1)
    covered = mask > 0

    alpha = cv2.GaussianBlur(mask, (0, 0), FEATHER_PX).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    out = frame.astype(np.float32) * (1.0 - alpha) + obscured.astype(np.float32) * alpha
    return out.round().astype(np.uint8), covered


def anonymize_stream(clip, stream, semantic_person, args):
    path = clip / stream
    if not path.is_file():
        return None

    frames, fps = read_video(path)
    sem = semantic_person if stream == SEMANTIC_SOURCE else None
    regions, counts = detect(
        frames, sem, args.yolo_weights, args.yolo_conf, args.face_thresh, args.imgsz
    )
    padded = pad_in_time(regions, TEMPORAL_PAD)

    out_frames, covered_masks = [], []
    for frame, region_list in zip(frames, padded):
        blurred, covered = blur_frame(frame, region_list)
        out_frames.append(blurred)
        covered_masks.append(covered)

    tmp = path.with_suffix(".anon.mp4")
    write_video(tmp, out_frames, fps, args.crf)
    tmp.replace(path)

    stats = {
        "stream": stream,
        "frames": len(frames),
        "frames_blurred": sum(bool(r) for r in padded),
        "regions": counts,
    }
    # Ground-truth check: of the person pixels the dataset itself labels,
    # how many ended up under the blur?
    if semantic_person is not None and stream == SEMANTIC_SOURCE:
        n = min(len(covered_masks), len(semantic_person))
        person_total = int(semantic_person[:n].sum())
        head_total = head_pixels = 0
        for i in range(n):
            for x1, y1, x2, y2 in mask_person_pixels(semantic_person[i]):
                sub = semantic_person[i, y1:y2, x1:x2]
                cut = max(1, round(float(y2 - y1) * HEAD_FRACTION))
                head = np.zeros_like(sub)
                head[:cut] = sub[:cut]
                head_total += int(head.sum())
                head_pixels += int((head & covered_masks[i][y1:y2, x1:x2]).sum())
        stats["semantic_check"] = {
            "person_frames": int((semantic_person[:n].reshape(n, -1).sum(1) > 0).sum()),
            "person_pixels": person_total,
            "head_pixels": head_total,
            "head_pixels_blurred": head_pixels,
            "head_recall": round(head_pixels / head_total, 4) if head_total else None,
        }
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory of clip folders.")
    parser.add_argument("--yolo_weights", default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--yolo_conf", type=float, default=0.10)
    parser.add_argument("--face_thresh", type=float, default=0.20,
                        help="CenterFace threshold. YOLO carries the recall here, "
                             "so lowering this mostly adds false positives.")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="YOLO inference size; upsampling helps small people.")
    parser.add_argument("--crf", type=int, default=17,
                        help="x264 quality for the rewritten streams (lower = better).")
    args = parser.parse_args(argv)

    clips = sorted(p for p in args.root.iterdir() if p.is_dir())
    if not clips:
        sys.exit(f"No clip directories under {args.root}")

    for clip in clips:
        print(f"\n=== {clip.name}")
        semantic_person = load_semantic_person(clip)
        report = []
        for stream in RGB_STREAMS:
            stats = anonymize_stream(clip, stream, semantic_person, args)
            if stats is None:
                continue
            report.append(stats)
            c = stats["regions"]
            print(f"  {stream:<18} {stats['frames_blurred']:>3}/{stats['frames']} frames "
                  f"blurred  (yolo {c['yolo']}, semantic {c['semantic']}, face {c['face']})")
            check = stats.get("semantic_check")
            if check and check["head_recall"] is not None:
                print(f"  {'':<18} semantic head-pixel recall "
                      f"{check['head_recall']:.1%} over {check['person_frames']} person-frames")

        meta_path = clip / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            meta["anonymization"] = {
                "method": "person-head blur (mosaic + gaussian)",
                "detectors": {
                    # Basename only -- meta.json is published, local paths are not.
                    "yolo": {"weights": Path(args.yolo_weights).name,
                             "conf": args.yolo_conf, "imgsz": args.imgsz},
                    "semantic": {"class": "ACTOR_PERSON", "streams": [SEMANTIC_SOURCE]},
                    "centerface": {"threshold": args.face_thresh, "in_shape": "2x native"},
                },
                "head_fraction": HEAD_FRACTION,
                "whole_body_below_px": WHOLE_BODY_BELOW_PX,
                "temporal_pad_frames": TEMPORAL_PAD,
                "streams": report,
                "untouched": ["route.mp4"],
                "note": "route.mp4 is a synthetic map render and contains no camera imagery.",
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
