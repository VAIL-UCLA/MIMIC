"""Sidecar I/O for action labels.

Each clip carries a label file beside it. Two schemas are understood, and both
may be stored as ``.npy``, ``.npz`` or ``.json``:

``poses``
    ``(N, 3)`` ego poses ``(x, y, yaw)`` in a clip-global frame, one per frame.
    Preferred — everything else is derivable from it exactly.

``waypoints``
    ``(N, K, 3)`` per-frame future waypoints in each frame's own ego frame,
    MIMIC's native label format. A global path is reconstructed from these when
    no poses are present, which is approximate — see
    :func:`~mimic.action_augmentation.trajectory.poses_from_waypoints`.

A bare ``.npy`` holding an array is read positionally: ``(N, C)`` with ``C >= 3``
is poses, of which only the leading ``(x, y, yaw)`` columns are used — any
trailing columns are derived quantities and are ignored. ``(N, K, 3)`` is
waypoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import trajectory as tj

#: Sidecar suffixes tried, in order, when looking beside a video.
SIDECAR_SUFFIXES = (".npy", ".npz", ".json")

#: Frame rate assumed when a sidecar carries no timestamps.
DEFAULT_FPS = 5.0


class LabelData:
    """A clip's trajectory: global ``poses`` ``(N, 3)`` and per-frame ``times`` ``(N,)``."""

    def __init__(
        self,
        poses: np.ndarray,
        times: np.ndarray,
        source: Path | None = None,
        waypoints: np.ndarray | None = None,
        label_times: np.ndarray | None = None,
        reconstructed: bool = False,
    ):
        self.poses = np.asarray(poses, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.source = source
        self.waypoints = waypoints
        self.label_times = (
            np.asarray(label_times, dtype=np.float64)
            if label_times is not None
            else tj.DEFAULT_LABEL_TIMES
        )
        #: True when poses were chained from waypoints rather than stored directly.
        self.reconstructed = reconstructed

        if self.poses.ndim != 2 or self.poses.shape[1] != 3:
            raise ValueError(f"poses must be (N, 3), got {self.poses.shape}")
        if self.times.shape != (len(self.poses),):
            raise ValueError(
                f"times must be ({len(self.poses)},), got {self.times.shape}"
            )

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0


def find_sidecar(video_path: Path, explicit: Path | None = None) -> Path:
    """Locate a clip's label file. Raises if nothing is found."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Label sidecar not found: {p}")
        return p

    video_path = Path(video_path)
    tried = []
    for suffix in SIDECAR_SUFFIXES:
        candidate = video_path.with_suffix(suffix)
        tried.append(candidate.name)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No label sidecar beside {video_path.name} (tried {', '.join(tried)}). "
        "Pass --labels to point at one explicitly."
    )


def _times_for(n: int, stored: np.ndarray | None, fps: float) -> np.ndarray:
    if stored is not None:
        times = np.asarray(stored, dtype=np.float64).ravel()
        if len(times) != n:
            raise ValueError(f"times has {len(times)} entries but there are {n} poses")
        return times
    return np.arange(n, dtype=np.float64) / fps


def load_labels(
    path: Path,
    fps: float = DEFAULT_FPS,
) -> LabelData:
    """Read a label sidecar into :class:`LabelData`."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path) as fh:
            raw = json.load(fh)
        fields = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items() if isinstance(v, list)}
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            fields = {k: data[k] for k in data.files}
    elif suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim == 2 and array.shape[1] >= 3:
            # Recorded pose files often carry derived columns after the pose —
            # e.g. the sample clips store (x, y, yaw, v, w). Take the pose.
            fields = {"poses": array[:, :3]}
        elif array.ndim == 3 and array.shape[2] == 3:
            fields = {"waypoints": array}
        else:
            raise ValueError(
                f"{path.name}: expected (N, >=3) poses or (N, K, 3) waypoints, "
                f"got {array.shape}"
            )
    else:
        raise ValueError(f"Unsupported sidecar format: {path.suffix}")

    stored_times = fields.get("times")
    label_times = fields.get("label_times")

    if "poses" in fields:
        poses = np.asarray(fields["poses"], dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"{path.name}: poses must be (N, 3), got {poses.shape}")
        times = _times_for(len(poses), stored_times, fps)
        return LabelData(
            poses, times, source=path,
            waypoints=fields.get("waypoints"), label_times=label_times,
        )

    if "waypoints" in fields:
        waypoints = np.asarray(fields["waypoints"], dtype=np.float64)
        if waypoints.ndim != 3 or waypoints.shape[2] != 3:
            raise ValueError(
                f"{path.name}: waypoints must be (N, K, 3), got {waypoints.shape}"
            )
        times = _times_for(len(waypoints), stored_times, fps)
        poses = tj.poses_from_waypoints(waypoints, times)
        return LabelData(
            poses, times, source=path, waypoints=waypoints,
            label_times=label_times, reconstructed=True,
        )

    raise ValueError(
        f"{path.name}: no 'poses' or 'waypoints' field (found: {', '.join(sorted(fields)) or 'nothing'})"
    )


def save_labels(
    path: Path,
    poses: np.ndarray,
    times: np.ndarray,
    waypoints: np.ndarray,
    label_times: np.ndarray,
    metadata: dict | None = None,
) -> Path:
    """Write an augmented label sidecar. Format follows ``path``'s suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    payload = {
        "poses": np.asarray(poses, dtype=np.float64),
        "times": np.asarray(times, dtype=np.float64),
        "waypoints": np.asarray(waypoints, dtype=np.float64),
        "label_times": np.asarray(label_times, dtype=np.float64),
    }

    if suffix == ".json":
        out = {k: v.tolist() for k, v in payload.items()}
        if metadata:
            out["metadata"] = metadata
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
    elif suffix in (".npz", ".npy"):
        # .npy cannot hold multiple arrays; write .npz and say so.
        if suffix == ".npy":
            path = path.with_suffix(".npz")
        if metadata:
            payload["metadata"] = np.array(json.dumps(metadata))
        np.savez(path, **payload)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
    return path
