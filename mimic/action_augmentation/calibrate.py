"""Recover the metric scale of a monocular depth reconstruction from action labels.

The depth model behind view synthesis (DepthCrafter) is relative, not metric: a
pixel's predicted depth ``Z_rel`` relates to its true depth by an unknown
constant, ``Z_rel = scale * Z_metric``. Rendering a lateral offset expressed in
meters therefore needs that constant, and guessing it silently mis-sizes every
generated maneuver.

The action labels already carry the answer. They record how far the robot
actually travelled between frames, in meters. Given that motion and the relative
depth, the scale is the value that makes the two agree — the *oracle* scale, in
the sense that it is read off ground truth rather than tuned by eye.

Method
------
For a static scene point seen at pixel ``p`` in frame A with relative depth
``Z_rel``, its metric position in camera A is ``(Z_rel / s) * ray``. Applying the
label-derived rigid motion ``(R, t)`` into camera B:

    X_B = (R * Z_rel * ray + s * t) / s

Projection ignores the positive factor ``1/s``, so with ``Y = R * Z_rel * ray``
the observed pixel ``p'`` in frame B must satisfy

    p'_h  x  (K Y + s K t) = 0

which is **linear in s**. Every tracked point contributes such a constraint, and
the least-squares solution over all of them is the scale. No small-angle
approximation and no assumption about the rotation — the label's yaw is applied
exactly.

Frames where the robot barely moves carry almost no information (``t -> 0``
leaves ``s`` unconstrained), so pairs with insufficient baseline are skipped, and
the per-pair estimates are combined with a median rather than a mean.
"""

from __future__ import annotations

import numpy as np

from .trajectory import relative_pose, wrap_angle

#: Intrinsics TrajectoryCrafter hardcodes in ``get_poses``. Calibration must use
#: the same ones as the renderer, not the physical camera's, or the scale will be
#: consistent with reality but wrong for the thing it feeds.
DEFAULT_FOCAL = 500.0
DEFAULT_CX = 512.0
DEFAULT_CY = 288.0

#: Robot-frame (x fwd, y left, z up) to camera-frame (x right, y down, z fwd).
ROBOT_TO_CAMERA = np.array(
    [
        [0.0, -1.0, 0.0],  # cam x  = -robot y   (right = -left)
        [0.0, 0.0, -1.0],  # cam y  = -robot z   (down  = -up)
        [1.0, 0.0, 0.0],   # cam z  =  robot x   (fwd   =  fwd)
    ]
)

#: Pairs with less baseline than this (meters) are dropped as uninformative.
MIN_BASELINE_M = 0.01


def intrinsics(focal: float = DEFAULT_FOCAL, cx: float = DEFAULT_CX, cy: float = DEFAULT_CY) -> np.ndarray:
    """3x3 pinhole intrinsic matrix."""
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_relative_transform(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid motion taking a point from camera A's frame into camera B's frame.

    Args:
        pose_a, pose_b: ``(x, y, yaw)`` robot poses in a shared global frame,
            meters and radians.

    Returns:
        ``(R, t)`` with ``X_B = R @ X_A + t``, in the camera convention
        (x right, y down, z forward). ``t`` is metric.
    """
    rel = relative_pose(np.asarray(pose_a, dtype=np.float64), np.asarray(pose_b, dtype=np.float64))
    dx, dy, dpsi = float(rel[0]), float(rel[1]), float(wrap_angle(rel[2]))

    # In robot coordinates: X_rB = Rz(-dpsi) @ (X_rA - [dx, dy, 0]).
    r_robot = _rot_z(-dpsi)
    m = ROBOT_TO_CAMERA
    rotation = m @ r_robot @ m.T
    translation = -m @ r_robot @ np.array([dx, dy, 0.0])
    return rotation, translation


def solve_scale(
    points_a: np.ndarray,
    points_b: np.ndarray,
    depth_rel: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    K: np.ndarray,
    trim: float = 0.2,
) -> tuple[float, int]:
    """Least-squares scale for one frame pair.

    Args:
        points_a: ``(M, 2)`` pixel coordinates in frame A.
        points_b: ``(M, 2)`` the same features tracked into frame B.
        depth_rel: ``(M,)`` relative depth at ``points_a``.
        rotation, translation: from :func:`camera_relative_transform`.
        K: 3x3 intrinsics.
        trim: Drop this fraction of worst-residual points before the final
            solve, so a few bad tracks do not drag the estimate.

    Returns:
        ``(scale, n_used)``. ``scale`` is NaN when the pair is degenerate.
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    depth_rel = np.asarray(depth_rel, dtype=np.float64)
    if len(points_a) == 0:
        return float("nan"), 0

    valid = np.isfinite(depth_rel) & (depth_rel > 0)
    valid &= np.isfinite(points_a).all(axis=1) & np.isfinite(points_b).all(axis=1)
    if valid.sum() < 3:
        return float("nan"), int(valid.sum())

    pa, pb, z = points_a[valid], points_b[valid], depth_rel[valid]

    k_inv = np.linalg.inv(K)
    rays = (k_inv @ np.hstack([pa, np.ones((len(pa), 1))]).T).T   # (M, 3)
    y = (rotation @ (z[:, None] * rays).T).T                      # (M, 3)

    ky = (K @ y.T).T                                              # (M, 3)
    kt = K @ translation                                          # (3,)
    pb_h = np.hstack([pb, np.ones((len(pb), 1))])                 # (M, 3)

    a = np.cross(pb_h, ky)                       # (M, 3)
    b = np.cross(pb_h, np.broadcast_to(kt, pb_h.shape))

    def _solve(a_, b_):
        denom = float((b_ * b_).sum())
        if denom < 1e-12:
            return float("nan")
        return -float((b_ * a_).sum()) / denom

    scale = _solve(a, b)
    if not np.isfinite(scale):
        return float("nan"), int(valid.sum())

    if trim > 0.0 and len(a) >= 8:
        residual = np.linalg.norm(a + scale * b, axis=1)
        keep = residual <= np.quantile(residual, 1.0 - trim)
        if keep.sum() >= 3:
            refined = _solve(a[keep], b[keep])
            if np.isfinite(refined):
                return refined, int(keep.sum())
    return scale, int(valid.sum())


def robust_scale(estimates: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """Combine per-pair estimates into one number, with spread diagnostics.

    The median is the reported value; ``mad_ratio`` (median absolute deviation
    over the median) says how much the pairs disagree. A ratio above ~0.2 means
    the estimate should not be trusted without looking at why.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    finite = np.isfinite(estimates) & (estimates > 0)
    if finite.sum() == 0:
        return {"scale": float("nan"), "median": float("nan"), "mad": float("nan"),
                "mad_ratio": float("nan"), "n_pairs": 0, "p16": float("nan"), "p84": float("nan")}

    vals = estimates[finite]
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[finite]
        order = np.argsort(vals)
        vals_sorted, w_sorted = vals[order], w[order]
        cum = np.cumsum(w_sorted)
        median = float(np.interp(0.5 * cum[-1], cum, vals_sorted))
    else:
        median = float(np.median(vals))

    mad = float(np.median(np.abs(vals - median)))
    return {
        "scale": median,
        "median": median,
        "mad": mad,
        "mad_ratio": float(mad / median) if median > 0 else float("nan"),
        "n_pairs": int(finite.sum()),
        "p16": float(np.percentile(vals, 16)),
        "p84": float(np.percentile(vals, 84)),
    }


def calibrate_from_tracks(
    tracks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    poses: np.ndarray,
    pair_indices: list[tuple[int, int]],
    K: np.ndarray | None = None,
    min_baseline: float = MIN_BASELINE_M,
    trim: float = 0.2,
) -> dict:
    """Scale from pre-computed feature tracks — the GPU-free core.

    Args:
        tracks: One ``(points_a, points_b, depth_rel)`` per pair.
        poses: ``(N, 3)`` robot poses, indexed by ``pair_indices``.
        pair_indices: ``(i, j)`` frame indices for each entry in ``tracks``.
        K: Intrinsics; defaults to the renderer's.
        min_baseline: Skip pairs whose metric translation is shorter than this.
        trim: Residual trimming fraction, see :func:`solve_scale`.

    Returns:
        The :func:`robust_scale` dict plus ``per_pair`` and ``skipped``.
    """
    K = intrinsics() if K is None else np.asarray(K, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)

    per_pair, counts, skipped = [], [], 0
    for (i, j), (pa, pb, z) in zip(pair_indices, tracks):
        rotation, translation = camera_relative_transform(poses[i], poses[j])
        if np.linalg.norm(translation) < min_baseline:
            skipped += 1
            per_pair.append(float("nan"))
            counts.append(0)
            continue
        scale, n = solve_scale(pa, pb, z, rotation, translation, K, trim=trim)
        per_pair.append(scale)
        counts.append(n)

    out = robust_scale(np.array(per_pair), np.array(counts, dtype=np.float64))
    out["per_pair"] = np.array(per_pair)
    out["skipped"] = skipped
    return out


# =====================================================================
# Affine-in-disparity model
# =====================================================================
# DepthCrafter is affine-invariant in *disparity*, and upstream's post-processing
# (normalize across the clip, then invert) preserves that:
#
#     1 / Z_rel = alpha' / Z_metric + beta
#
# Only when beta == 0 does a single multiplicative scale hold exactly. Otherwise
# the meters-per-depth-unit conversion varies with depth, and any single --scale
# is a local linearization. Fitting both parameters says how far off that is.

#: Range of beta searched, in units of 1/Z_rel. Scene-dependent but generous.
_BETA_SEARCH = 241


def _depth_warp(depth_rel: np.ndarray, beta: float) -> np.ndarray:
    """``Z_rel -> w`` such that ``Z_metric = alpha * w`` under the affine model."""
    denom = 1.0 - beta * depth_rel
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(denom) > 1e-9, depth_rel / denom, np.nan)
    return w


def reprojection_error(
    points_a: np.ndarray,
    points_b: np.ndarray,
    warped_depth: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    K: np.ndarray,
    alpha: float,
) -> float:
    """Median reprojection error in **pixels** for a candidate fit.

    The cross-product residual used to *solve* for the scale is proportional to
    the depths it is built from, so it shrinks whenever a candidate warps depths
    smaller — comparing it across different ``beta`` rewards shrinking rather
    than fitting. Pixel error has no such dependence and is what to compare.
    """
    valid = np.isfinite(warped_depth) & (warped_depth > 0)
    if valid.sum() < 3 or not np.isfinite(alpha):
        return float("inf")
    k_inv = np.linalg.inv(K)
    rays = (k_inv @ np.hstack([points_a[valid], np.ones((valid.sum(), 1))]).T).T
    y = (rotation @ (warped_depth[valid][:, None] * rays).T).T
    x_b = alpha * y + translation

    ahead = x_b[:, 2] > 1e-6
    if ahead.sum() < 3:
        return float("inf")
    projected = (K @ x_b[ahead].T).T
    predicted = projected[:, :2] / projected[:, 2:3]
    return float(np.median(np.linalg.norm(predicted - points_b[valid][ahead], axis=1)))


def _residual_for(pa, pb, w, rotation, translation, K, trim):
    """Solve the linear scale for pre-warped depths, scored in pixel error."""
    scale, n = solve_scale(pa, pb, w, rotation, translation, K, trim=trim)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan"), float("inf"), 0
    err = reprojection_error(pa, pb, w, rotation, translation, K, 1.0 / scale)
    return scale, err, n


def solve_affine(
    points_a: np.ndarray,
    points_b: np.ndarray,
    depth_rel: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    K: np.ndarray,
    beta_range: tuple[float, float] | None = None,
    trim: float = 0.2,
) -> dict:
    """Fit ``1/Z_rel = alpha'/Z_metric + beta`` for one frame pair.

    ``alpha`` is solved in closed form for each candidate ``beta`` (the constraint
    stays linear once depths are warped), so this is a 1-D search rather than a
    joint optimization.

    Returns:
        ``alpha`` (metric depth per warped unit), ``beta``, ``scale`` (the
        equivalent pure-scale fit, ``beta = 0``), and the median reprojection
        residual of each.
    """
    depth_rel = np.asarray(depth_rel, dtype=np.float64)
    finite = depth_rel[np.isfinite(depth_rel) & (depth_rel > 0)]
    if len(finite) < 3:
        return {"alpha": float("nan"), "beta": float("nan"), "scale": float("nan"),
                "residual": float("inf"), "residual_scale_only": float("inf"), "n": 0}

    # beta must keep 1 - beta*Z_rel positive for every point, so bound it below 1/max.
    limit = 1.0 / float(finite.max())
    lo, hi = beta_range if beta_range is not None else (-4.0 * limit, 0.98 * limit)

    scale_only, res_only, n_only = _residual_for(
        points_a, points_b, depth_rel, rotation, translation, K, trim
    )

    best = {"alpha": float("nan"), "beta": 0.0, "residual": res_only, "n": n_only}
    for beta in np.linspace(lo, hi, _BETA_SEARCH):
        w = _depth_warp(depth_rel, beta)
        s, res, n = _residual_for(points_a, points_b, w, rotation, translation, K, trim)
        if np.isfinite(res) and res < best["residual"]:
            best = {"alpha": 1.0 / s, "beta": float(beta), "residual": res, "n": n}

    if not np.isfinite(best["alpha"]) and np.isfinite(scale_only):
        best["alpha"] = 1.0 / scale_only
    return {
        "alpha": best["alpha"],
        "beta": best["beta"],
        "scale": scale_only,
        "residual": best["residual"],
        "residual_scale_only": res_only,
        "n": best["n"],
    }


def effective_scale(alpha: float, beta: float, depth_rel_ref: float) -> float:
    """Local depth-units-per-meter at a reference depth, under the affine model.

    ``Z_metric = alpha * Z_rel / (1 - beta * Z_rel)``, so

        d Z_rel / d Z_metric = (1 - beta * Z_rel)^2 / alpha

    With ``beta = 0`` this collapses to ``1 / alpha``, the global scale.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        return float("nan")
    return float((1.0 - beta * depth_rel_ref) ** 2 / alpha)
