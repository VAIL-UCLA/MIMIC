"""Calibrate a depth scale per clip and store it beside each one.

Each clip's recorded poses say exactly how far the robot moved between frames,
which pins the conversion from the depth reconstruction's units to meters —
the *oracle* scale for that clip. This walks a set of clips, solves that scale
against each one's own labels, and writes it to a ``.scale.json`` sidecar that
:mod:`.augment_action` picks up automatically.

Why per clip rather than once for the corpus: DepthCrafter normalizes disparity
over the frames it is given, so its units are set by the depth range present in
that particular window (see :mod:`.scales`). The same camera and the same model
produce a different scale on a long street than in a narrow alley.

Usage::

    # one clip
    python -m mimic.action_augmentation.calibrate_clips --input clip.mp4

    # a corpus, caching depth so a re-run costs no GPU time
    python -m mimic.action_augmentation.calibrate_clips \\
        --input 'assets/clips/*/rgb_pinhole.mp4' --fps 20

    # reuse depth stacks computed elsewhere, no GPU at all
    python -m mimic.action_augmentation.calibrate_clips \\
        --input 'clips/*.mp4' --depth_cache depths/ --no_recompute

The window calibrated is the leading ``--length`` frames at ``--stride``, which
is exactly what TrajectoryCrafter's reader consumes, so the stored number
applies to the render that follows.
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
from . import scales as scale_io
from .calibrate_scale import (
    RENDER_HEIGHT,
    RENDER_WIDTH,
    calibrate_clip,
    estimate_depth,
    load_depth,
    read_frames,
)

#: Default place for cached depth stacks. Gitignored — they are large and
#: reproducible, so they are a cache, not an artifact.
DEFAULT_DEPTH_CACHE = Path(".depth_cache")

#: Fields of a calibration result worth keeping in the sidecar. The rest is
#: per-pair detail that belongs in the aggregate report, not beside the clip.
KEEP = (
    "n_pairs", "skipped", "n_frames", "mad_ratio", "p16", "p84",
    "residual_scale_only", "alpha", "beta", "depth_p10", "depth_p50",
    "depth_p90", "effective_scale", "depth_scale_spread",
)


def cache_path(cache_dir: Path, video_path: Path, window: dict) -> Path:
    """Where a clip's depth stack is cached.

    The window is part of the name: a stack computed over a different set of
    frames was normalized differently and is not interchangeable.
    """
    tag = f"{window['start']}_{window['length']}_{window['stride']}"
    return cache_dir / f"{video_path.stem}__{tag}.npy"


def depth_for_clip(video_path, window, args, tc_root):
    """Depth stack for a clip: from cache when possible, else DepthCrafter.

    Returns ``(depth, source)`` where source names where it came from.
    """
    cache = None
    if args.depth_cache is not None:
        cache = cache_path(Path(args.depth_cache), video_path, window)
        if cache.is_file() and not args.recompute:
            return load_depth(cache, window["length"]), f"cache {cache.name}"

    if args.no_recompute:
        raise FileNotFoundError(
            f"No cached depth for {video_path.name} and --no_recompute is set"
            + (f" (looked for {cache})" if cache else "")
        )
    if tc_root is None:
        raise FileNotFoundError("DepthCrafter is needed but TrajectoryCrafter was not found")

    depth = estimate_depth(
        video_path, window["length"], window["stride"], tc_root, args.device
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # float16 halves a ~115 MB stack; the scale solve is nowhere near
        # that precision-limited.
        np.save(cache, depth.astype(np.float16))
    return depth, "DepthCrafter"


def calibrate_one(video_path: Path, args, tc_root) -> dict:
    """Calibrate a single clip and write its sidecar."""
    window = scale_io.make_window(args.length, args.stride)

    sidecar = label_io.find_sidecar(video_path, Path(args.labels) if args.labels else None)
    data = label_io.load_labels(sidecar, fps=args.fps or label_io.DEFAULT_FPS)

    frames, src_fps = read_frames(video_path, args.length, args.stride)
    fps = (args.fps or (src_fps if src_fps > 0 else label_io.DEFAULT_FPS)) / args.stride

    depth, depth_source = depth_for_clip(video_path, window, args, tc_root)

    result = calibrate_clip(
        video_path, depth, data, fps, frames,
        pair_gap=args.pair_gap, min_baseline=args.min_baseline,
        affine=not args.no_affine,
    )
    result["depth_source"] = depth_source
    result["labels"] = sidecar.name
    result["fps"] = fps

    if np.isfinite(result.get("scale", np.nan)):
        stats = {k: result[k] for k in KEEP if k in result and np.isfinite(result[k])}
        stats["depth_source"] = depth_source
        stats["fps"] = fps
        out_path = None
        if args.scale_dir:
            out_path = Path(args.scale_dir) / scale_io.scale_path(video_path).name
        written = scale_io.save_scale(
            video_path, result["scale"], window,
            labels=sidecar, calibration=stats, path=out_path,
        )
        result["sidecar"] = str(written)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a per-clip depth scale from recorded poses and store it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Video file, or a glob for several (quote it).")
    parser.add_argument("--labels", type=str, default=None,
                        help="Label sidecar. Default: found beside each video.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate. Default: read from the video.")
    parser.add_argument("--length", type=int, default=49,
                        help="Frames in the render window, matching TrajectoryCrafter.")
    parser.add_argument("--stride", type=int, default=1, help="Frame sampling stride.")
    parser.add_argument("--depth_cache", type=str, default=str(DEFAULT_DEPTH_CACHE),
                        help="Directory for cached depth stacks. Empty string disables.")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cached depth and run DepthCrafter again.")
    parser.add_argument("--no_recompute", action="store_true",
                        help="Fail rather than run DepthCrafter. Use with a warm cache.")
    parser.add_argument("--scale_dir", type=str, default=None,
                        help="Write sidecars here instead of beside each clip.")
    parser.add_argument("--tc_root", type=str, default=None,
                        help="TrajectoryCrafter checkout. Default: the third_party submodule.")
    parser.add_argument("--pair_gap", type=int, default=1,
                        help="Frames between the two views of a pair. Raise it for slow clips.")
    parser.add_argument("--min_baseline", type=float, default=cal.MIN_BASELINE_M,
                        help="Skip pairs whose metric translation is shorter than this (m).")
    parser.add_argument("--no_affine", action="store_true",
                        help="Skip the affine-in-disparity fit.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default=None,
                        help="Write the full multi-clip report as JSON.")
    args = parser.parse_args(argv)
    if args.depth_cache == "":
        args.depth_cache = None
    return args


def _row(result: dict) -> str:
    name = result.get("clip", "?")
    if result.get("error"):
        return f"  {name:<28} ERROR  {result['error']}"
    drift = result.get("depth_scale_spread", float("nan"))
    return (
        f"  {name:<28} {result['scale']:>9.4f}  "
        f"{result['mad_ratio'] * 100:>6.1f}%  "
        f"{result.get('residual_scale_only', float('nan')):>7.3f}  "
        f"{drift * 100 if np.isfinite(drift) else float('nan'):>7.1f}%  "
        f"{result['n_pairs']:>5}"
    )


def _summarize(results: list[dict]) -> None:
    scales = [r["scale"] for r in results if np.isfinite(r.get("scale", np.nan))]
    if not scales:
        return
    stats = cal.robust_scale(np.array(scales))
    lo, hi = min(scales), max(scales)
    print("\n" + "-" * 72)
    print(f"  {len(scales)} clip(s) calibrated.  median {stats['scale']:.4f}, "
          f"range {lo:.4f} .. {hi:.4f}")
    if stats["scale"] > 0:
        spread = (hi - lo) / stats["scale"]
        print(f"  clip-to-clip spread {spread * 100:.1f}% of the median.")
        if spread > 0.15:
            print("    -> the clips genuinely disagree; a single shared --scale would")
            print("       mis-size maneuvers on the clips furthest from the median.")
        else:
            print("    -> these clips happen to agree closely; a shared scale would")
            print("       have been an acceptable approximation for this set.")
    print("-" * 72)
    print("  Each clip's scale is stored beside it; augment_action reads it by default.")


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
    if args.labels and len(paths) > 1:
        print("error: --labels names one sidecar but several clips matched --input",
              file=sys.stderr)
        return 1

    tc_root = None
    if not args.no_recompute:
        try:
            from .augment_action import resolve_tc_root

            tc_root = resolve_tc_root(args.tc_root)
        except FileNotFoundError as exc:
            # Only fatal once a clip actually needs depth computed.
            print(f"note: {exc}", file=sys.stderr)

    print(f"Calibrating {len(paths)} clip(s) at {RENDER_WIDTH}x{RENDER_HEIGHT}, "
          f"window {args.length}@stride {args.stride}\n")
    print(f"  {'clip':<28} {'scale':>9}  {'MAD':>6}  {'reproj':>7}  {'drift':>7}  {'pairs':>5}")

    results = []
    for path in paths:
        try:
            result = calibrate_one(path, args, tc_root)
        except (OSError, ValueError, FileNotFoundError, RuntimeError) as exc:
            result = {"clip": path.name, "error": str(exc), "scale": float("nan"),
                      "n_pairs": 0}
        results.append(result)
        print(_row(result))

    _summarize(results)

    if args.out:
        payload = {
            "window": scale_io.make_window(args.length, args.stride),
            "clips": [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in r.items() if k != "per_pair"}
                for r in results
            ],
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  Report written to {out}")

    return 0 if any(np.isfinite(r.get("scale", np.nan)) for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
