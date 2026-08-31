"""Per-clip depth scale sidecars.

The depth reconstruction behind the view synthesis is not metric, so
:mod:`.augment_action` needs a factor converting a lateral offset in meters
into depth units. That factor is **not** a constant of the camera. DepthCrafter
normalizes its disparity over the frames it is handed::

    depths = (res - res.min()) / (res.max() - res.min())    # models/infer.py
    depths = 10000.0 / (depths * 3900)

``res.min()`` and ``res.max()`` are taken over the whole batch, so the units are
fixed by the depth range that happens to be present in *that* window of *that*
clip. A clip looking down a long street and a clip in a narrow alley get
different normalizations, hence different scales. Calibrating once and reusing
the number across a corpus is only valid to the extent the clips happen to
share a depth range.

So each clip carries its own scale, read off its recorded poses by
:mod:`.calibrate_clips` and stored beside it::

    clip.mp4
    clip.npy          <- action labels
    clip.scale.json   <- this

A scale is only valid for the window it was calibrated on. The renderer always
consumes the leading ``video_length`` frames at a fixed stride, so that is the
window that gets calibrated, and the record stores it so a mismatch is caught
rather than silently applied.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Suffix of the sidecar written beside each clip.
SCALE_SUFFIX = ".scale.json"

#: Bumped when the record layout changes incompatibly.
SCHEMA_VERSION = 1

#: Sentinel accepted by ``--scale`` meaning "read the sidecar".
AUTO = "auto"


class ScaleNotFound(FileNotFoundError):
    """No sidecar beside the clip, and no explicit scale given."""


def scale_path(video_path: Path | str) -> Path:
    """Where a clip's scale sidecar lives."""
    return Path(video_path).with_suffix(SCALE_SUFFIX)


def make_window(length: int, stride: int = 1) -> dict:
    """Describe the frame window a scale applies to.

    ``start`` is always 0: TrajectoryCrafter's reader takes the leading frames
    of a clip, so that is the only window the renderer ever sees.
    """
    return {"start": 0, "length": int(length), "stride": int(stride)}


def save_scale(
    video_path: Path | str,
    scale: float,
    window: dict,
    labels: Path | str | None = None,
    calibration: dict | None = None,
    path: Path | str | None = None,
) -> Path:
    """Write a clip's scale sidecar. Returns the path written."""
    video_path = Path(video_path)
    out = Path(path) if path is not None else scale_path(video_path)
    record = {
        "version": SCHEMA_VERSION,
        "clip": video_path.name,
        "scale": float(scale),
        "units": "depth units per meter",
        "window": dict(window),
        "labels": Path(labels).name if labels is not None else None,
    }
    if calibration is not None:
        record["calibration"] = calibration
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return out


def load_scale(video_path: Path | str, path: Path | str | None = None) -> dict | None:
    """Read a clip's scale sidecar, or ``None`` if there isn't one."""
    src = Path(path) if path is not None else scale_path(video_path)
    if not src.is_file():
        return None
    with open(src) as fh:
        record = json.load(fh)
    if "scale" not in record:
        raise ValueError(f"{src.name}: no 'scale' field")
    version = record.get("version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{src.name}: schema version {version}, but this build understands "
            f"up to {SCHEMA_VERSION}"
        )
    return record


def window_mismatch(record: dict, window: dict) -> str | None:
    """Describe how a record's window differs from the one about to be rendered.

    Returns ``None`` when they agree. The scale depends on the frames the depth
    model normalized over, so rendering a different window than was calibrated
    means the number no longer applies.
    """
    stored = record.get("window")
    if not stored:
        return "the sidecar records no window"
    differing = [
        f"{k}: calibrated {stored.get(k)!r}, rendering {window.get(k)!r}"
        for k in ("start", "length", "stride")
        if stored.get(k) != window.get(k)
    ]
    return "; ".join(differing) if differing else None


def resolve_scale(
    video_path: Path | str,
    requested: float | str | None = AUTO,
    window: dict | None = None,
    path: Path | str | None = None,
) -> tuple[float, str]:
    """Settle on the scale to render with.

    ``requested`` may be a number, which wins outright, or :data:`AUTO` (the
    default), which reads the sidecar. Returns ``(scale, provenance)`` where
    provenance is a short phrase naming where the number came from, suitable
    for logging next to the render.

    Raises :class:`ScaleNotFound` when ``AUTO`` finds no sidecar — rendering at
    an unverified scale silently mis-sizes every generated maneuver, which is
    the failure this whole path exists to prevent.
    """
    video_path = Path(video_path)

    if requested is not None and requested != AUTO:
        try:
            value = float(requested)
        except (TypeError, ValueError):
            raise ValueError(
                f"--scale must be a number or {AUTO!r}, got {requested!r}"
            ) from None
        if not (value > 0.0):
            raise ValueError(f"--scale must be positive, got {value}")
        return value, "given on the command line"

    record = load_scale(video_path, path)
    if record is None:
        raise ScaleNotFound(
            f"No scale sidecar for {video_path.name} (looked for "
            f"{scale_path(video_path).name}). Calibrate it against the clip's "
            f"recorded poses:\n"
            f"    python -m mimic.action_augmentation.calibrate_clips "
            f"--input {video_path}\n"
            f"or pass an explicit --scale if you already know it."
        )

    scale = float(record["scale"])
    provenance = f"{scale_path(video_path).name}"
    if window is not None:
        mismatch = window_mismatch(record, window)
        if mismatch:
            provenance += f" (WARNING: window mismatch -- {mismatch})"
    return scale, provenance
