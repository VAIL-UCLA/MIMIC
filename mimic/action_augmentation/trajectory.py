"""Trajectory math for corrective behavior expansion.

All poses are SE(2) in MIMIC's convention: ``x`` forward, ``y`` left, ``yaw`` in
radians counter-clockwise, positions in meters.

The augmentation builds a *deviate-and-recover* maneuver on top of a recorded
path. Over a 4 s horizon the robot drifts laterally to a peak offset by 2 s and
rejoins the recorded path by 4 s:

    offset
      ^
    s |       ___
      |     /     \\
      |   /         \\
    0 +--/-----------\\---->  t
      0      2 s      4 s

Only the *lateral* offset is specified. Heading follows from it — yaw at each
sample is the tangent of the resulting path, so the maneuver stays kinematically
consistent instead of the robot sliding sideways.

The new action label is that maneuver re-expressed in the ego frame at each
timestep, which is exactly what a policy must output to correct the drift it
sees in the generated video.
"""

from __future__ import annotations

import numpy as np

#: Supervision horizon in seconds. The deviation peaks at half of it.
DEFAULT_HORIZON_S = 4.0

#: MIMIC's label timestamps: 15 waypoints, 0.2 s – 5.0 s, non-uniform.
DEFAULT_LABEL_TIMES = np.array(
    [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0],
    dtype=np.float64,
)

PROFILES = ("raised_cosine", "smoothstep", "triangle")


# =====================================================================
# Offset profile
# =====================================================================


def lateral_offset_profile(
    t: np.ndarray,
    strength: float,
    horizon: float = DEFAULT_HORIZON_S,
    profile: str = "raised_cosine",
) -> np.ndarray:
    """Lateral offset in meters at times ``t``: 0 → ``strength`` → 0 over ``horizon``.

    Args:
        t: Times in seconds, any shape. Values outside ``[0, horizon]`` clamp to 0.
        strength: Peak lateral offset in meters. Positive is left (``+y``).
        horizon: Total maneuver duration; the peak lands at ``horizon / 2``.
        profile: ``raised_cosine`` (default, zero velocity at both ends and at the
            peak), ``smoothstep`` (zero velocity at the ends, faster mid-phase),
            or ``triangle`` (piecewise linear — discontinuous lateral velocity,
            useful only as a baseline).

    Returns:
        Offsets with the same shape as ``t``.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}. Available: {', '.join(PROFILES)}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    t = np.asarray(t, dtype=np.float64)
    u = np.clip(t / horizon, 0.0, 1.0)  # normalized time over the maneuver

    if profile == "raised_cosine":
        # (1 - cos 2*pi*u)/2: 0 at u=0 and u=1, 1 at u=0.5, zero slope at all three.
        shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * u))
    elif profile == "smoothstep":
        # Smoothstep out to the peak, mirrored back. Zero slope at the ends.
        v = np.where(u <= 0.5, u * 2.0, (1.0 - u) * 2.0)
        shape = v * v * (3.0 - 2.0 * v)
    else:  # triangle
        shape = np.where(u <= 0.5, u * 2.0, (1.0 - u) * 2.0)

    # Outside the maneuver window the robot is back on the recorded path.
    shape = np.where((t < 0.0) | (t > horizon), 0.0, shape)
    return strength * shape


# =====================================================================
# SE(2) helpers
# =====================================================================


def se2_matrix(pose: np.ndarray) -> np.ndarray:
    """``(x, y, yaw)`` → 3×3 homogeneous transform. Accepts a batch ``(..., 3)``."""
    pose = np.asarray(pose, dtype=np.float64)
    c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    m = np.zeros(pose.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0], m[..., 0, 1], m[..., 0, 2] = c, -s, pose[..., 0]
    m[..., 1, 0], m[..., 1, 1], m[..., 1, 2] = s, c, pose[..., 1]
    m[..., 2, 2] = 1.0
    return m


def se2_inverse(pose: np.ndarray) -> np.ndarray:
    """Inverse of an ``(x, y, yaw)`` pose. Accepts a batch ``(..., 3)``."""
    pose = np.asarray(pose, dtype=np.float64)
    c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    x, y = pose[..., 0], pose[..., 1]
    out = np.empty_like(pose)
    out[..., 0] = -(c * x + s * y)
    out[..., 1] = -(-s * x + c * y)
    out[..., 2] = -pose[..., 2]
    return out


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[-pi, pi)``.

    Note the half-open end: exactly ``+pi`` maps to ``-pi``. They are the same
    heading, so this only matters if you compare wrapped values for equality.
    """
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def relative_pose(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Express ``target`` poses in the frame of ``reference``.

    Args:
        reference: ``(3,)`` pose defining the frame.
        target: ``(..., 3)`` poses in the same global frame as ``reference``.

    Returns:
        ``(..., 3)`` poses relative to ``reference``.
    """
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    c, s = np.cos(reference[2]), np.sin(reference[2])
    dx = target[..., 0] - reference[0]
    dy = target[..., 1] - reference[1]
    out = np.empty_like(target)
    out[..., 0] = c * dx + s * dy
    out[..., 1] = -s * dx + c * dy
    out[..., 2] = wrap_angle(target[..., 2] - reference[2])
    return out


# =====================================================================
# Path construction
# =====================================================================


def path_tangent_yaw(
    positions: np.ndarray,
    fallback: np.ndarray | None = None,
    trust: np.ndarray | None = None,
) -> np.ndarray:
    """Heading at each sample from the path tangent, via central differences.

    Samples where the robot is essentially stationary have no meaningful tangent;
    those fall back to ``fallback`` (the recorded heading) when given, and
    otherwise hold the previous valid heading.

    ``trust`` marks samples whose tangent the caller knows to be unreliable even
    though the step is nonzero. Displacing a *parked* robot sideways produces a
    perfectly good tangent that points 90 degrees away from where the robot is
    actually facing, so the caller has to say which samples had real motion of
    their own — see :func:`apply_lateral_offset`.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n < 2:
        return np.zeros(n) if fallback is None else np.asarray(fallback, dtype=np.float64).copy()

    d = np.empty_like(positions)
    d[1:-1] = positions[2:] - positions[:-2]  # central, second-order
    if n >= 3:
        # Second-order one-sided differences at the ends. A plain first-order
        # difference biases the endpoint heading by half the step, which shows up
        # as a few degrees of spurious yaw at t=0 where the maneuver should still
        # be straight.
        d[0] = -3.0 * positions[0] + 4.0 * positions[1] - positions[2]
        d[-1] = 3.0 * positions[-1] - 4.0 * positions[-2] + positions[-3]
    else:
        d[0] = positions[1] - positions[0]
        d[-1] = positions[-1] - positions[-2]

    step = np.hypot(d[:, 0], d[:, 1])
    yaw = np.arctan2(d[:, 1], d[:, 0])

    # Degenerate tangents: too small a step to trust the direction.
    bad = step < 1e-9
    if trust is not None:
        bad = bad | ~np.asarray(trust, dtype=bool)
    if bad.any():
        if fallback is not None:
            yaw[bad] = np.asarray(fallback, dtype=np.float64)[bad]
        else:
            for i in np.flatnonzero(bad):
                yaw[i] = yaw[i - 1] if i > 0 else 0.0
    return wrap_angle(yaw)


def path_normals(poses: np.ndarray) -> np.ndarray:
    """Unit left-normal at each pose, i.e. the ``+y`` direction of its own frame."""
    yaw = np.asarray(poses, dtype=np.float64)[:, 2]
    return np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)


#: How far the recorded path must move, relative to the offset it is being
#: displaced by, before its tangent is taken as the new heading. Below this the
#: recorded heading is held instead. The ratio bounds the heading a maneuver can
#: manufacture out of near-stationary footage at ``atan(1 / ratio)`` — 26.6
#: degrees at 2.0, comfortably above the ~15 degrees a real maneuver produces at
#: walking pace.
TANGENT_DOMINANCE = 2.0

#: Absolute floor, in meters, on the recorded central-difference step. Below
#: this the robot is parked and there is no heading to recover from its path at
#: all — not even where the offset profile is momentarily flat, which is exactly
#: where the dominance ratio alone degenerates into 0 >= 0.
MIN_TRAVEL_M = 1e-3


def apply_lateral_offset(
    poses: np.ndarray,
    offsets: np.ndarray,
    recompute_yaw: bool = True,
    dominance: float = TANGENT_DOMINANCE,
) -> np.ndarray:
    """Displace a path sideways by a per-sample offset.

    Each pose moves along its own left-normal, so the deviation is perpendicular
    to the recorded heading and the construction works on curved paths too.

    Args:
        poses: ``(N, 3)`` recorded ego poses in a clip-global frame.
        offsets: ``(N,)`` lateral offsets in meters, positive left.
        recompute_yaw: Take heading from the tangent of the displaced path. This
            is what keeps the maneuver kinematically consistent; disable it only
            to inspect a pure sideways translation.
        dominance: How much the recorded motion must exceed the offset's own
            motion before the displaced tangent is trusted — see
            :data:`TANGENT_DOMINANCE`. Recorded clips routinely sit at a
            crossing for seconds at a time, and a parked robot pushed sideways
            has a tangent perpendicular to the way it is facing.

    Returns:
        ``(N, 3)`` displaced poses.
    """
    poses = np.asarray(poses, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"poses must be (N, 3), got {poses.shape}")
    if offsets.shape != (len(poses),):
        raise ValueError(f"offsets must be ({len(poses)},), got {offsets.shape}")

    out = poses.copy()
    out[:, :2] = poses[:, :2] + offsets[:, None] * path_normals(poses)
    if recompute_yaw:
        out[:, 2] = _yaw_with_offset(poses, out[:, :2], offsets, dominance)
    return out


def _yaw_with_offset(
    poses: np.ndarray, displaced: np.ndarray, offsets: np.ndarray, dominance: float
) -> np.ndarray:
    """Recorded heading plus the turn the lateral offset actually induces.

    Taking the displaced path's tangent as the heading outright is wrong on
    recorded footage in two ways. A robot backing up has a path tangent 180
    degrees from the way it faces, and a parked robot displaced sideways gets a
    tangent perpendicular to it — the sample clips do both, sitting at crossings
    and reversing at up to 2.2 m/s. Both artifacts are properties of the
    *recorded* path, so both cancel in the difference between the displaced and
    recorded tangents, which is the only part the maneuver contributes.

    Where even that difference is meaningless — the robot barely moved, so the
    offset's own sideways rate is all there is — the delta is dropped and the
    recorded heading stands.
    """
    tangent_recorded = path_tangent_yaw(poses[:, :2])
    tangent_displaced = path_tangent_yaw(displaced)
    delta = wrap_angle(tangent_displaced - tangent_recorded)
    delta[~_tangent_is_trustworthy(poses[:, :2], offsets, dominance)] = 0.0
    return wrap_angle(poses[:, 2] + delta)


def _central_step(values: np.ndarray) -> np.ndarray:
    """Magnitude of the central difference used by :func:`path_tangent_yaw`."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return np.zeros(n)
    d = np.empty_like(values)
    d[1:-1] = values[2:] - values[:-2]
    if n >= 3:
        d[0] = -3.0 * values[0] + 4.0 * values[1] - values[2]
        d[-1] = 3.0 * values[-1] - 4.0 * values[-2] + values[-3]
    else:
        d[0] = values[1] - values[0]
        d[-1] = values[-1] - values[-2]
    return np.abs(d) if d.ndim == 1 else np.hypot(d[:, 0], d[:, 1])


def _tangent_is_trustworthy(
    positions: np.ndarray, offsets: np.ndarray, dominance: float
) -> np.ndarray:
    """Where the recorded motion outweighs the sideways displacement.

    The displaced path's tangent is a mix of the robot's real travel and the
    offset profile's sideways rate. Where the robot barely moved, the second
    term is all that is left and the tangent stops describing a heading.
    """
    travelled = _central_step(np.asarray(positions, dtype=np.float64))
    sideways = _central_step(np.asarray(offsets, dtype=np.float64))
    return (travelled > MIN_TRAVEL_M) & (travelled >= dominance * sideways)


#: Below this the robot counts as stopped when looking for somewhere to put a
#: maneuver. Delivery robots idle at crossings for seconds at a time.
DEFAULT_MIN_SPEED_MPS = 0.2


def path_speed(poses: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Speed at each sample, in meters per second, from the path itself."""
    poses = np.asarray(poses, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(poses) < 2:
        return np.zeros(len(poses))
    dp = np.gradient(poses[:, :2], axis=0)
    dt = np.gradient(times)
    dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
    return np.hypot(dp[:, 0], dp[:, 1]) / dt


#: How far a recorded heading may sit from its own path tangent before the
#: sample is treated as corrupt. Recorded yaw is usually derived from the path,
#: so a large disagreement means a bad sample rather than a sharp turn — a 180
#: degree flip at a wrap boundary is the common one.
YAW_CONSISTENCY_TOL_RAD = np.pi / 6.0


def yaw_is_consistent(
    poses: np.ndarray,
    times: np.ndarray,
    min_speed: float = DEFAULT_MIN_SPEED_MPS,
    tol: float = YAW_CONSISTENCY_TOL_RAD,
) -> np.ndarray:
    """Samples whose recorded heading agrees with the direction of travel.

    A single bad heading is quiet in the recorded data — nothing moves, the
    pose is where it should be — but a maneuver displaces along the heading's
    normal, so one flipped sample throws the path a full ``2 * offset`` sideways
    and manufactures an enormous turn out of nothing.

    Samples slower than ``min_speed`` have no direction of travel to compare
    against and are reported as consistent; use :func:`path_speed` to exclude
    them separately.
    """
    poses = np.asarray(poses, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(poses) < 3:
        return np.ones(len(poses), dtype=bool)
    moving = path_speed(poses, times) > min_speed
    delta = wrap_angle(path_tangent_yaw(poses[:, :2]) - poses[:, 2])
    return ~(moving & (np.abs(delta) > tol))


def best_maneuver_start(
    poses: np.ndarray,
    times: np.ndarray,
    duration: float,
    min_speed: float = DEFAULT_MIN_SPEED_MPS,
) -> tuple[float, float]:
    """Earliest start time whose ``duration`` window is the cleanest to use.

    A maneuver is a deviation from where the robot was *going*. Run one over a
    stretch where it was parked at a crossing and the label says to slide
    sideways and back while the video does not move — a sample that teaches the
    policy nothing true. Run one across a corrupt heading and the label contains
    a turn that never happened. This finds somewhere better to put it.

    Returns ``(start_time, usable_fraction)``, the fraction being samples in the
    chosen window that are both moving and self-consistent, so a caller can
    refuse or warn when even the best window is poor.
    """
    times = np.asarray(times, dtype=np.float64)
    if len(times) == 0:
        return 0.0, 0.0
    moving = (path_speed(poses, times) > min_speed) & yaw_is_consistent(
        poses, times, min_speed
    )

    best_start, best_fraction = float(times[0]), -1.0
    for i, start in enumerate(times):
        window = (times >= start) & (times <= start + duration)
        if not window.any():
            continue
        # Stop once the window runs off the end of the clip, or the tail would
        # win on a short sample rather than on actually moving.
        if times[-1] - start < duration and best_fraction >= 0.0:
            break
        fraction = float(moving[window].mean())
        if fraction > best_fraction:
            best_start, best_fraction = float(start), fraction
        if best_fraction >= 1.0:
            break
        del i
    return best_start, max(best_fraction, 0.0)


def deviate_and_recover(
    poses: np.ndarray,
    times: np.ndarray,
    strength: float,
    horizon: float = DEFAULT_HORIZON_S,
    start_time: float = 0.0,
    profile: str = "raised_cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the deviate-and-recover path from a recorded one.

    Args:
        poses: ``(N, 3)`` recorded ego poses in a clip-global frame.
        times: ``(N,)`` timestamps in seconds, matching ``poses``.
        strength: Peak lateral offset in meters. Positive is left.
        horizon: Maneuver duration; the peak lands at ``horizon / 2``.
        start_time: When the maneuver begins, in the same clock as ``times``.
        profile: Offset profile — see :func:`lateral_offset_profile`.

    Returns:
        ``(new_poses, offsets)``.
    """
    times = np.asarray(times, dtype=np.float64)
    offsets = lateral_offset_profile(times - start_time, strength, horizon, profile)
    return apply_lateral_offset(poses, offsets), offsets


# =====================================================================
# Label generation
# =====================================================================


def waypoints_from_path(
    poses: np.ndarray,
    times: np.ndarray,
    label_times: np.ndarray = DEFAULT_LABEL_TIMES,
    clamp_tail: bool = True,
) -> np.ndarray:
    """Per-frame future waypoints in each frame's own ego frame.

    For frame *i* at time ``t_i``, the path is sampled at ``t_i + label_times``
    and expressed relative to pose *i* — MIMIC's label format.

    Args:
        poses: ``(N, 3)`` poses in a clip-global frame.
        times: ``(N,)`` timestamps in seconds.
        label_times: ``(K,)`` future offsets in seconds.
        clamp_tail: Near the end of the clip the horizon runs past the last
            sample. When True those queries clamp to the final pose; when False
            they are NaN, so a caller can drop the incomplete tail.

    Returns:
        ``(N, K, 3)`` waypoints as ``(x, y, yaw)`` in each frame's ego frame.
    """
    poses = np.asarray(poses, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    label_times = np.asarray(label_times, dtype=np.float64)
    if len(poses) != len(times):
        raise ValueError(f"poses ({len(poses)}) and times ({len(times)}) must match")
    if len(poses) < 2:
        raise ValueError("need at least 2 poses to interpolate a label")

    query = times[:, None] + label_times[None, :]  # (N, K)
    flat = query.ravel()

    # Linear interpolation in position; yaw is unwrapped first so interpolation
    # does not cut across the +/-pi branch.
    x = np.interp(flat, times, poses[:, 0])
    y = np.interp(flat, times, poses[:, 1])
    yaw = np.interp(flat, times, np.unwrap(poses[:, 2]))
    sampled = np.stack([x, y, yaw], axis=-1).reshape(query.shape + (3,))

    out = np.empty_like(sampled)
    for i in range(len(poses)):
        out[i] = relative_pose(poses[i], sampled[i])

    if not clamp_tail:
        out[query > times[-1]] = np.nan
    return out


def build_augmented_label(
    poses: np.ndarray,
    times: np.ndarray,
    strength: float,
    horizon: float = DEFAULT_HORIZON_S,
    start_time: float = 0.0,
    profile: str = "raised_cosine",
    label_times: np.ndarray = DEFAULT_LABEL_TIMES,
    mode: str = "deviate_recover",
) -> dict[str, np.ndarray]:
    """Full augmentation: displaced path plus its action label.

    Args:
        mode: ``deviate_recover`` builds the S-curve maneuver and labels it.
            ``reexpress`` holds a constant lateral offset instead and re-expresses
            the *recorded* path in the offset frame — a pure recovery label, with
            no deviation maneuver of its own.

    Returns:
        Dict with ``poses`` ``(N, 3)`` displaced path, ``offsets`` ``(N,)``,
        ``waypoints`` ``(N, K, 3)`` label, and ``label_times`` ``(K,)``.
    """
    poses = np.asarray(poses, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    if mode == "deviate_recover":
        new_poses, offsets = deviate_and_recover(
            poses, times, strength, horizon, start_time, profile
        )
        # The label is the maneuver itself: what the robot should do from here.
        waypoints = waypoints_from_path(new_poses, times, label_times)
    elif mode == "reexpress":
        # Camera sits at a fixed offset; the target stays the recorded path, so
        # the label points back toward it.
        offsets = np.full(len(poses), float(strength))
        new_poses = apply_lateral_offset(poses, offsets, recompute_yaw=False)
        recorded = waypoints_from_path(poses, times, label_times)
        waypoints = np.empty_like(recorded)
        for i in range(len(poses)):
            # Re-express frame i's recorded future in the displaced frame.
            global_wp = _to_global(poses[i], recorded[i])
            waypoints[i] = relative_pose(new_poses[i], global_wp)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Available: deviate_recover, reexpress")

    return {
        "poses": new_poses,
        "offsets": offsets,
        "waypoints": waypoints,
        "label_times": np.asarray(label_times, dtype=np.float64),
    }


def _to_global(reference: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Inverse of :func:`relative_pose` — lift ego-frame poses back to global."""
    reference = np.asarray(reference, dtype=np.float64)
    local = np.asarray(local, dtype=np.float64)
    c, s = np.cos(reference[2]), np.sin(reference[2])
    out = np.empty_like(local)
    out[..., 0] = reference[0] + c * local[..., 0] - s * local[..., 1]
    out[..., 1] = reference[1] + s * local[..., 0] + c * local[..., 1]
    out[..., 2] = wrap_angle(reference[2] + local[..., 2])
    return out


def poses_from_waypoints(waypoints: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Recover a clip-global path from per-frame ego-frame waypoints.

    A fallback for datasets that store only labels and not ego poses. Prefer a
    poses sidecar when you have one: this chains per-frame steps, and each step
    is a straight-line interpolation of the label, so on a curved path it cuts
    the chord and the error accumulates along the clip — on the order of
    centimetres over ten-odd seconds at walking pace, growing with curvature and
    clip length. Frame 0 is placed at the origin.

    The tail is also unreliable: once ``times[i] + label_times[-1]`` runs past
    the clip, the label saturates and the reconstruction falls short.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if waypoints.ndim != 3 or waypoints.shape[2] != 3:
        raise ValueError(f"waypoints must be (N, K, 3), got {waypoints.shape}")

    poses = np.zeros((len(waypoints), 3), dtype=np.float64)
    for i in range(1, len(waypoints)):
        dt = times[i] - times[i - 1]
        # Where frame i-1 expected to be after dt, taken as frame i's pose.
        step = _interp_waypoint(waypoints[i - 1], dt)
        poses[i] = _to_global(poses[i - 1], step)
    return poses


def _interp_waypoint(waypoints_k: np.ndarray, t: float, label_times: np.ndarray = DEFAULT_LABEL_TIMES) -> np.ndarray:
    """Sample one frame's waypoint set at time ``t`` (linear, clamped).

    The label starts at 0.2 s, but callers need shorter queries — one frame at
    20 fps is 0.05 s. The frame's own origin is the waypoint at t=0, so it is
    prepended; without it every sub-0.2 s query would clamp up to the first
    waypoint and the reconstructed path would run far too fast.
    """
    label_times = np.asarray(label_times, dtype=np.float64)[: len(waypoints_k)]
    ts = np.concatenate([[0.0], label_times])
    xs = np.concatenate([[0.0], waypoints_k[:, 0]])
    ys = np.concatenate([[0.0], waypoints_k[:, 1]])
    yaws = np.concatenate([[0.0], np.unwrap(waypoints_k[:, 2])])
    return np.array(
        [np.interp(t, ts, xs), np.interp(t, ts, ys), wrap_angle(np.interp(t, ts, yaws))],
        dtype=np.float64,
    )
