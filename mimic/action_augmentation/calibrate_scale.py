"""Recover the oracle depth scale for a clip from its action labels.

``--scale`` in :mod:`.augment_action` converts a lateral offset in meters into
the units of the monocular depth reconstruction. Guessing it silently mis-sizes
every generated maneuver. The action labels already record how far the robot
actually moved, so the scale can be read off ground truth instead — the *oracle*
scale.

The clip is read at the same 1024x576 the renderer uses, features are tracked
between frames, and each tracked point contributes a constraint that is linear
in the scale (see :mod:`.calibrate`). Per-pair estimates are combined with a
median so a few bad tracks or a stationary stretch cannot skew the result.

Usage:
    # depth computed with DepthCrafter, exactly as the renderer would
    python -m mimic.action_augmentation.calibrate_scale --input clip.mp4

    # reuse a precomputed depth stack (N, H, W) — no GPU needed
    python -m mimic.action_augmentation.calibrate_scale \\
        --input clip.mp4 --depth depth.npy

    # calibrate over a set of clips and report the corpus-level scale
    python -m mimic.action_augmentation.calibrate_scale \\
        --input 'data/*/front_*.mp4' --depth_dir depths/ --out scale.json

This reports one number across everything it is given, which is useful for
asking how much a corpus agrees. It is *not* a camera constant: DepthCrafter
normalizes disparity over the frames it is handed, so the units depend on the
clip. For the per-clip scales the renderer actually consumes, use
:mod:`.calibrate_clips`, which writes a sidecar beside each clip.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from . import calibrate as cal
from . import labels as label_io

#: The renderer reads every clip at this resolution and pairs it with the
#: intrinsics hardcoded in its get_poses, so calibration must match both.
RENDER_WIDTH = 1024
RENDER_HEIGHT = 576

#: Feature tracking defaults.
MAX_FEATURES = 1200
FB_ERROR_PX = 1.0


def read_frames(
    video_path: Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, float]:
    """Read a clip as grayscale at the renderer's resolution.

    Returns:
        ``(frames (N, H, W) uint8, source_fps)``.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame = cv2.resize(frame, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"{video_path.name}: need at least 2 frames, got {len(frames)}")
    return np.stack(frames), float(fps)


def track_features(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    max_features: int = MAX_FEATURES,
    fb_error_px: float = FB_ERROR_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Track corners from ``frame_a`` into ``frame_b``.

    Tracks are verified by running the flow backwards and keeping only those that
    land back within ``fb_error_px`` of where they started. Bad correspondences
    bias the scale far more than missing ones, so this errs toward rejecting.
    """
    import cv2

    corners = cv2.goodFeaturesToTrack(
        frame_a, maxCorners=max_features, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if corners is None or len(corners) < 3:
        return np.zeros((0, 2)), np.zeros((0, 2))

    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    fwd, status_f, _ = cv2.calcOpticalFlowPyrLK(frame_a, frame_b, corners, None, **lk)
    if fwd is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    back, status_b, _ = cv2.calcOpticalFlowPyrLK(frame_b, frame_a, fwd, None, **lk)
    if back is None:
        return np.zeros((0, 2)), np.zeros((0, 2))

    pa = corners.reshape(-1, 2)
    pb = fwd.reshape(-1, 2)
    fb = np.linalg.norm(back.reshape(-1, 2) - pa, axis=1)
    good = (
        (status_f.ravel() == 1)
        & (status_b.ravel() == 1)
        & (fb < fb_error_px)
        & np.isfinite(pb).all(axis=1)
    )
    return pa[good], pb[good]


def sample_depth(depth_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear depth lookup at subpixel feature locations."""
    h, w = depth_map.shape[:2]
    x = np.clip(points[:, 0], 0, w - 1.001)
    y = np.clip(points[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    d = depth_map
    return (
        d[y0, x0] * (1 - fx) * (1 - fy)
        + d[y0, x0 + 1] * fx * (1 - fy)
        + d[y0 + 1, x0] * (1 - fx) * fy
        + d[y0 + 1, x0 + 1] * fx * fy
    )


def load_depth(path: Path, n_frames: int) -> np.ndarray:
    """Load a precomputed depth stack and reshape it to ``(N, H, W)``."""
    depth = np.load(path, allow_pickle=False)
    depth = np.squeeze(depth)
    if depth.ndim == 2:
        depth = depth[None]
    if depth.ndim != 3:
        raise ValueError(f"{path.name}: expected (N, H, W) depth, got {depth.shape}")
    if len(depth) < n_frames:
        raise ValueError(
            f"{path.name}: has {len(depth)} depth maps but the clip has {n_frames} frames"
        )
    return depth[:n_frames].astype(np.float64)


def estimate_depth(video_path: Path, n_frames: int, stride: int, tc_root: Path, device: str) -> np.ndarray:
    """Run DepthCrafter through the TrajectoryCrafter submodule.

    Uses upstream's own reader and estimator so the depths are identical to the
    ones the renderer will see — calibrating against a different depth pipeline
    would give a consistent number for the wrong quantity.
    """
    import os

    import torch

    if str(tc_root) not in sys.path:
        sys.path.insert(0, str(tc_root))
    cwd = Path.cwd()
    os.chdir(tc_root)
    try:
        from models.infer import DepthCrafterDemo
        from models.utils import read_video_frames

        frames = read_video_frames(str(video_path), n_frames, stride, 1024)
        estimator = DepthCrafterDemo(
            unet_path="tencent/DepthCrafter",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload="model",
            device=device,
        )
        with torch.inference_mode():
            depths = estimator.infer(frames, 0.0001, 10000.0, 5, 1.0, window_size=110, overlap=25)
        return depths.squeeze(1).float().cpu().numpy().astype(np.float64)
    finally:
        os.chdir(cwd)


def poses_at_frame_times(data: label_io.LabelData, n_frames: int, fps: float) -> np.ndarray:
    """Interpolate the label's ego poses onto the frame timeline."""
    frame_times = np.arange(n_frames, dtype=np.float64) / fps
    if frame_times[-1] > data.times[-1] + 1e-6:
        frame_times = np.clip(frame_times, None, data.times[-1])
    x = np.interp(frame_times, data.times, data.poses[:, 0])
    y = np.interp(frame_times, data.times, data.poses[:, 1])
    yaw = np.interp(frame_times, data.times, np.unwrap(data.poses[:, 2]))
    return np.stack([x, y, yaw], axis=-1)


def calibrate_clip(
    video_path: Path,
    depth: np.ndarray,
    data: label_io.LabelData,
    fps: float,
    frames: np.ndarray,
    pair_gap: int = 1,
    min_baseline: float = cal.MIN_BASELINE_M,
    affine: bool = True,
) -> dict:
    """Full per-clip calibration from frames, depth and labels."""
    n = min(len(frames), len(depth))
    poses = poses_at_frame_times(data, n, fps)

    tracks, pairs = [], []
    for i in range(0, n - pair_gap, pair_gap):
        j = i + pair_gap
        pa, pb = track_features(frames[i], frames[j])
        if len(pa) < 8:
            continue
        z = sample_depth(depth[i], pa)
        tracks.append((pa, pb, z))
        pairs.append((i, j))

    if not tracks:
        return {"scale": float("nan"), "n_pairs": 0, "error": "no usable feature tracks"}

    out = cal.calibrate_from_tracks(tracks, poses, pairs, min_baseline=min_baseline)
    out["clip"] = video_path.name
    out["n_frames"] = n

    if affine:
        alphas, betas, res_a, res_s = [], [], [], []
        for (i, j), (pa, pb, z) in zip(pairs, tracks):
            R, t = cal.camera_relative_transform(poses[i], poses[j])
            if np.linalg.norm(t) < min_baseline:
                continue
            fit = cal.solve_affine(pa, pb, z, R, t, cal.intrinsics())
            if np.isfinite(fit["alpha"]):
                alphas.append(fit["alpha"])
                betas.append(fit["beta"])
                res_a.append(fit["residual"])
                res_s.append(fit["residual_scale_only"])
        if alphas:
            alpha = float(np.median(alphas))
            beta = float(np.median(betas))
            # Score across the depth range the features actually cover. The
            # centre pixel is a poor reference on a sidewalk — it lands near the
            # horizon, where depth is huge and barely constrained.
            tracked = np.concatenate([z for _, _, z in tracks])
            tracked = tracked[np.isfinite(tracked) & (tracked > 0)]
            z10, z50, z90 = (float(np.percentile(tracked, q)) for q in (10, 50, 90))
            s10 = cal.effective_scale(alpha, beta, z10)
            s50 = cal.effective_scale(alpha, beta, z50)
            s90 = cal.effective_scale(alpha, beta, z90)
            # How much the meters-per-unit conversion drifts over that range.
            spread = abs(s90 - s10) / s50 if np.isfinite(s50) and s50 > 0 else float("nan")
            out.update(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "depth_p10": z10,
                    "depth_p50": z50,
                    "depth_p90": z90,
                    "effective_scale": s50,
                    "effective_scale_near": s10,
                    "effective_scale_far": s90,
                    "depth_scale_spread": float(spread),
                    "residual_affine": float(np.median(res_a)),
                    "residual_scale_only": float(np.median(res_s)),
                }
            )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover the oracle depth scale for --scale from a clip's action labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Video file, or a glob for several (quote it).")
    parser.add_argument("--labels", type=str, default=None,
                        help="Label sidecar. Default: found beside each video.")
    parser.add_argument("--depth", type=str, default=None,
                        help="Precomputed depth stack (N, H, W) .npy. Skips DepthCrafter.")
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Directory of <stem>.npy depth stacks, for multi-clip runs.")
    parser.add_argument("--tc_root", type=str, default=None,
                        help="TrajectoryCrafter checkout. Default: the third_party submodule.")
    parser.add_argument("--out", type=str, default=None, help="Write the full report as JSON.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate. Default: read from the video, else the label rate.")
    parser.add_argument("--max_frames", type=int, default=49,
                        help="Frames per clip; matches the renderer's window by default.")
    parser.add_argument("--stride", type=int, default=1, help="Frame sampling stride.")
    parser.add_argument("--pair_gap", type=int, default=1,
                        help="Frames between the two views of a pair. Raise it for slow clips "
                             "where consecutive frames have too little baseline.")
    parser.add_argument("--min_baseline", type=float, default=cal.MIN_BASELINE_M,
                        help="Skip pairs whose metric translation is shorter than this (m).")
    parser.add_argument("--no_affine", action="store_true",
                        help="Skip the affine-in-disparity fit and report only a single scale.")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args(argv)


def _report(result: dict) -> None:
    print(f"\n  {result.get('clip', '?')}")
    if result.get("error"):
        print(f"    error: {result['error']}")
        return
    print(f"    pairs used     : {result['n_pairs']} ({result.get('skipped', 0)} skipped, low baseline)")
    print(f"    scale          : {result['scale']:.4f} depth units / meter")
    print(f"    spread (MAD)   : {result['mad_ratio'] * 100:.1f}%  "
          f"[p16 {result['p16']:.3f} .. p84 {result['p84']:.3f}]")
    print(f"    reproj error   : {result.get('residual_scale_only', float('nan')):.3f} px")
    if "alpha" in result:
        print(f"    depth range    : {result['depth_p10']:.1f} .. {result['depth_p90']:.1f} "
              f"(median {result['depth_p50']:.1f})")
        print(f"    affine fit     : alpha {result['alpha']:.4f}, beta {result['beta']:+.5f}")
        spread = result["depth_scale_spread"]
        if np.isfinite(spread):
            print(f"    scale drift    : {spread * 100:.1f}% across that range "
                  f"({result['effective_scale_near']:.3f} near .. "
                  f"{result['effective_scale_far']:.3f} far)")
            if spread > 0.25:
                print("      -> depth is affine, not proportional: one --scale is a compromise.")
                print("         It will be right mid-scene and off at the extremes.")
            else:
                print("      -> a single --scale fits this clip.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    paths = [Path(p).expanduser().resolve() for p in sorted(glob.glob(args.input))]
    if not paths:
        single = Path(args.input).expanduser().resolve()
        if single.is_file():
            paths = [single]
    if not paths:
        print(f"error: no videos matched {args.input!r}", file=sys.stderr)
        return 1

    tc_root = None
    if args.depth is None and args.depth_dir is None:
        try:
            from .augment_action import resolve_tc_root

            tc_root = resolve_tc_root(args.tc_root)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(f"Calibrating over {len(paths)} clip(s), {RENDER_WIDTH}x{RENDER_HEIGHT}, "
          f"pair_gap={args.pair_gap}")

    results, scales = [], []
    for path in paths:
        try:
            sidecar = label_io.find_sidecar(path, Path(args.labels) if args.labels else None)
            data = label_io.load_labels(sidecar, fps=args.fps or label_io.DEFAULT_FPS)
            frames, src_fps = read_frames(path, args.max_frames, args.stride)
            fps = args.fps or (src_fps if src_fps > 0 else label_io.DEFAULT_FPS)
            fps = fps / args.stride

            if args.depth is not None:
                depth = load_depth(Path(args.depth), len(frames))
            elif args.depth_dir is not None:
                depth = load_depth(Path(args.depth_dir) / f"{path.stem}.npy", len(frames))
            else:
                depth = estimate_depth(path, args.max_frames, args.stride, tc_root, args.device)

            result = calibrate_clip(
                path, depth, data, fps, frames,
                pair_gap=args.pair_gap, min_baseline=args.min_baseline,
                affine=not args.no_affine,
            )
        except (OSError, ValueError, FileNotFoundError) as exc:
            result = {"clip": path.name, "error": str(exc), "scale": float("nan"), "n_pairs": 0}

        results.append(result)
        _report(result)
        if np.isfinite(result.get("scale", np.nan)):
            scales.append(result["scale"])

    if not scales:
        print("\nNo clip produced a usable scale.", file=sys.stderr)
        return 1

    summary = cal.robust_scale(np.array(scales))
    print("\n" + "=" * 58)
    print(f"  ORACLE SCALE : {summary['scale']:.4f} depth units / meter")
    print(f"  across {summary['n_pairs']} clip(s), spread {summary['mad_ratio'] * 100:.1f}%")
    print("=" * 58)
    if summary["mad_ratio"] > 0.2:
        print("  WARNING: clips disagree by more than 20%. Check that the labels are")
        print("           metric and time-aligned with the video before trusting this.")
    print(f"\n  Use it with:  --scale {summary['scale']:.4f}")

    if args.out:
        payload = {
            "oracle_scale": summary["scale"],
            "summary": {k: v for k, v in summary.items() if k != "per_pair"},
            "clips": [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in r.items() if k != "per_pair"}
                for r in results
            ],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  Report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
