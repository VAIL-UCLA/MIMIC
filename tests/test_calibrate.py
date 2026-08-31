"""Tests for oracle scale recovery.

Ground truth is synthetic: 3D points are projected exactly under a known camera
motion at a known scale, so the estimator must return that scale. This is what
validates the robot->camera sign conventions — get any axis backwards and the
recovered scale comes out wrong or negative.
"""

import numpy as np
import pytest

from mimic.action_augmentation import calibrate as cal
from mimic.action_augmentation import trajectory as tj

K = cal.intrinsics()


def project(points_cam, K=K):
    """Pinhole projection; drops points at or behind the image plane."""
    z = points_cam[:, 2]
    ok = z > 1e-6
    uv = (K @ points_cam[ok].T).T
    return uv[:, :2] / uv[:, 2:3], ok


def synth_scene(n=400, seed=0, z_range=(3.0, 25.0), spread=8.0):
    """Random metric 3D points in front of the camera."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(*z_range, size=n)
    x = rng.uniform(-spread, spread, size=n)
    y = rng.uniform(-spread / 2, spread / 2, size=n)
    return np.stack([x, y, z], axis=-1)


def make_pair(pose_a, pose_b, scale, n=400, seed=0, noise_px=0.0):
    """Exact correspondences between two views, with relative depth at `scale`."""
    pts_metric = synth_scene(n=n, seed=seed)
    pa, ok_a = project(pts_metric)
    pts_metric = pts_metric[ok_a]

    R, t = cal.camera_relative_transform(pose_a, pose_b)
    pts_b = (R @ pts_metric.T).T + t
    pb, ok_b = project(pts_b)

    pa, pts_metric = pa[ok_b], pts_metric[ok_b]
    if noise_px > 0:
        pb = pb + np.random.default_rng(seed + 99).normal(0, noise_px, pb.shape)
    # The depth model reports scale * metric depth.
    depth_rel = scale * pts_metric[:, 2]
    return pa, pb, depth_rel, R, t


# ── the transform itself ─────────────────────────────────────────────

def test_identity_motion_is_identity():
    R, t = cal.camera_relative_transform([0, 0, 0], [0, 0, 0])
    assert np.allclose(R, np.eye(3), atol=1e-12)
    assert np.allclose(t, 0.0, atol=1e-12)


def test_forward_motion_moves_points_closer():
    """Robot drives 1 m forward -> static points lose 1 m of camera z."""
    R, t = cal.camera_relative_transform([0, 0, 0], [1, 0, 0])
    X = np.array([0.0, 0.0, 10.0])
    assert (R @ X + t)[2] == pytest.approx(9.0)


def test_left_motion_pushes_points_right():
    """Robot steps 1 m left -> the world appears 1 m to the right (camera +x)."""
    R, t = cal.camera_relative_transform([0, 0, 0], [0, 1, 0])
    X = np.array([0.0, 0.0, 10.0])
    assert (R @ X + t)[0] == pytest.approx(1.0)


def test_yaw_left_rotates_scene_right():
    """Turning left by 90 deg puts a point that was ahead onto the camera's right."""
    R, t = cal.camera_relative_transform([0, 0, 0], [0, 0, np.pi / 2])
    out = R @ np.array([0.0, 0.0, 10.0]) + t
    assert out[0] == pytest.approx(10.0)
    assert out[2] == pytest.approx(0.0, abs=1e-9)


def test_transform_composes_with_relative_pose():
    """Round-trip: A->B then B->A returns the original point."""
    a, b = np.array([1.0, -2.0, 0.4]), np.array([3.5, 0.7, -0.3])
    R1, t1 = cal.camera_relative_transform(a, b)
    R2, t2 = cal.camera_relative_transform(b, a)
    X = np.array([1.2, -0.4, 8.0])
    assert np.allclose(R2 @ (R1 @ X + t1) + t2, X, atol=1e-9)


# ── scale recovery on exact data ─────────────────────────────────────

@pytest.mark.parametrize("scale", [0.05, 0.5, 1.0, 3.7, 20.0])
def test_recovers_scale_exactly(scale):
    pose_a, pose_b = np.array([0.0, 0, 0]), np.array([0.4, 0, 0])
    pa, pb, z, R, t = make_pair(pose_a, pose_b, scale)
    got, n = cal.solve_scale(pa, pb, z, R, t, K)
    assert n > 100
    assert got == pytest.approx(scale, rel=1e-9)


@pytest.mark.parametrize(
    "pose_b",
    [
        [0.4, 0.0, 0.0],       # straight ahead
        [0.4, 0.15, 0.0],      # forward + left
        [0.4, 0.0, 0.10],      # forward + yaw
        [0.3, -0.2, -0.15],    # forward + right + yaw right
        [0.0, 0.3, 0.0],       # pure lateral
    ],
)
def test_recovers_scale_under_varied_motion(pose_b):
    """Rotation is handled exactly, so yaw must not bias the estimate."""
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array(pose_b, dtype=float), 2.5)
    got, _ = cal.solve_scale(pa, pb, z, R, t, K)
    assert got == pytest.approx(2.5, rel=1e-8)


def test_scale_is_independent_of_baseline_length():
    """A longer baseline is better conditioned but must not change the answer."""
    got = []
    for d in (0.05, 0.2, 1.0, 3.0):
        pa, pb, z, R, t = make_pair(np.zeros(3), np.array([d, 0, 0]), 1.7)
        got.append(cal.solve_scale(pa, pb, z, R, t, K)[0])
    assert np.allclose(got, 1.7, rtol=1e-8)


def test_zero_baseline_is_degenerate():
    """No translation means no depth information; must report NaN, not a number."""
    pa, pb, z, R, t = make_pair(np.zeros(3), np.zeros(3), 1.0)
    got, _ = cal.solve_scale(pa, pb, z, R, t, K)
    assert np.isnan(got)


# ── robustness ───────────────────────────────────────────────────────

def test_tolerates_tracking_noise():
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array([0.5, 0, 0]), 2.0, n=800, noise_px=0.5)
    got, _ = cal.solve_scale(pa, pb, z, R, t, K)
    assert got == pytest.approx(2.0, rel=0.05)


def test_trimming_survives_outlier_tracks():
    """A tenth of the correspondences are garbage; the estimate should hold."""
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array([0.5, 0, 0]), 2.0, n=600)
    rng = np.random.default_rng(3)
    bad = rng.choice(len(pb), size=len(pb) // 10, replace=False)
    pb = pb.copy()
    pb[bad] += rng.normal(0, 60, (len(bad), 2))
    trimmed, _ = cal.solve_scale(pa, pb, z, R, t, K, trim=0.25)
    untrimmed, _ = cal.solve_scale(pa, pb, z, R, t, K, trim=0.0)
    assert abs(trimmed - 2.0) < abs(untrimmed - 2.0)
    assert trimmed == pytest.approx(2.0, rel=0.1)


def test_invalid_depths_are_dropped():
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array([0.5, 0, 0]), 2.0, n=300)
    z = z.copy()
    z[:50] = np.nan
    z[50:80] = -1.0
    got, n = cal.solve_scale(pa, pb, z, R, t, K)
    assert n <= len(z) - 80
    assert got == pytest.approx(2.0, rel=1e-8)


def test_too_few_points_is_nan():
    got, _ = cal.solve_scale(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2),
                             np.eye(3), np.array([0.5, 0, 0]), K)
    assert np.isnan(got)


def test_empty_input_is_nan():
    got, n = cal.solve_scale(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0),
                             np.eye(3), np.array([1.0, 0, 0]), K)
    assert np.isnan(got) and n == 0


# ── aggregation ──────────────────────────────────────────────────────

def test_robust_scale_ignores_outlier_pairs():
    est = np.array([2.0, 2.02, 1.98, 2.01, 50.0, np.nan, -3.0])
    out = cal.robust_scale(est)
    assert out["scale"] == pytest.approx(2.01, abs=0.05)
    assert out["n_pairs"] == 5          # NaN and the negative are excluded


def test_mad_ratio_flags_disagreement():
    tight = cal.robust_scale(np.array([2.0, 2.01, 1.99, 2.02]))
    loose = cal.robust_scale(np.array([1.0, 2.0, 3.0, 4.0]))
    assert tight["mad_ratio"] < 0.02
    assert loose["mad_ratio"] > 0.2


def test_robust_scale_all_invalid():
    out = cal.robust_scale(np.array([np.nan, np.nan]))
    assert np.isnan(out["scale"]) and out["n_pairs"] == 0


# ── end to end over a clip ───────────────────────────────────────────

def test_calibrate_from_tracks_over_a_clip():
    """A moving clip: every pair should agree on the same scale."""
    scale = 3.2
    times = np.arange(30) / 5.0
    poses = np.zeros((30, 3))
    poses[:, 0] = times * 1.2                       # 1.2 m/s forward
    poses[:, 1] = 0.3 * np.sin(times * 0.8)         # gentle weave
    poses[:, 2] = tj.path_tangent_yaw(poses[:, :2])

    tracks, pairs = [], []
    for i in range(len(poses) - 1):
        pa, pb, z, _, _ = make_pair(poses[i], poses[i + 1], scale, n=300, seed=i)
        tracks.append((pa, pb, z))
        pairs.append((i, i + 1))

    out = cal.calibrate_from_tracks(tracks, poses, pairs)
    assert out["scale"] == pytest.approx(scale, rel=1e-6)
    assert out["n_pairs"] == len(pairs)
    assert out["mad_ratio"] < 1e-6


def test_stationary_pairs_are_skipped_not_averaged_in():
    """A parked stretch carries no scale information and must be dropped."""
    scale = 2.0
    poses = np.zeros((20, 3))
    poses[:10, 0] = np.arange(10) * 0.25            # moving
    poses[10:, 0] = poses[9, 0]                     # parked

    tracks, pairs = [], []
    for i in range(len(poses) - 1):
        pa, pb, z, _, _ = make_pair(poses[i], poses[i + 1], scale, n=200, seed=i)
        tracks.append((pa, pb, z))
        pairs.append((i, i + 1))

    out = cal.calibrate_from_tracks(tracks, poses, pairs)
    assert out["skipped"] == 10                     # the parked pairs
    assert out["scale"] == pytest.approx(scale, rel=1e-6)


# ── affine-in-disparity model ────────────────────────────────────────
# DepthCrafter is affine-invariant in disparity, and upstream normalizes then
# inverts, so 1/Z_rel = alpha'/Z_metric + beta. A pure scale is the beta == 0
# special case.

def make_pair_affine(pose_a, pose_b, alpha, beta, n=500, seed=0):
    """Correspondences whose depth follows the affine-in-disparity model."""
    pts = synth_scene(n=n, seed=seed)
    pa, ok_a = project(pts)
    pts = pts[ok_a]
    R, t = cal.camera_relative_transform(pose_a, pose_b)
    pb, ok_b = project((R @ pts.T).T + t)
    pa, pts = pa[ok_b], pts[ok_b]
    # Z_metric = alpha * Z_rel / (1 - beta * Z_rel)  =>  Z_rel = Z_m / (alpha + beta*Z_m)
    z_metric = pts[:, 2]
    depth_rel = z_metric / (alpha + beta * z_metric)
    return pa, pb, depth_rel, R, t


def test_affine_reduces_to_pure_scale_when_beta_zero():
    pa, pb, z, R, t = make_pair_affine(np.zeros(3), np.array([0.5, 0, 0]), alpha=0.4, beta=0.0)
    out = cal.solve_affine(pa, pb, z, R, t, K)
    assert out["scale"] == pytest.approx(1 / 0.4, rel=1e-6)
    assert abs(out["beta"]) < 1e-3
    assert out["residual_scale_only"] < 1e-6


def test_affine_recovers_a_depth_offset():
    """With beta != 0 no single scale fits; the two-parameter model still does."""
    alpha, beta = 0.4, 0.03
    pa, pb, z, R, t = make_pair_affine(np.zeros(3), np.array([0.5, 0, 0]), alpha, beta)
    out = cal.solve_affine(pa, pb, z, R, t, K)
    assert out["residual"] < out["residual_scale_only"]
    assert out["beta"] == pytest.approx(beta, abs=0.01)
    assert out["alpha"] == pytest.approx(alpha, rel=0.15)


def test_single_scale_is_biased_when_beta_nonzero():
    """The headline number to report: how wrong a single scale is."""
    alpha, beta = 0.4, 0.03
    pa, pb, z, R, t = make_pair_affine(np.zeros(3), np.array([0.5, 0, 0]), alpha, beta)
    out = cal.solve_affine(pa, pb, z, R, t, K)
    # A pure-scale fit leaves visible reprojection residual where affine does not.
    assert out["residual_scale_only"] > 1e-4
    assert out["residual"] < out["residual_scale_only"] / 2


def test_effective_scale_matches_pure_scale_when_beta_zero():
    assert cal.effective_scale(alpha=0.4, beta=0.0, depth_rel_ref=12.0) == pytest.approx(2.5)


def test_effective_scale_varies_with_depth_when_beta_nonzero():
    near = cal.effective_scale(alpha=0.4, beta=0.02, depth_rel_ref=5.0)
    far = cal.effective_scale(alpha=0.4, beta=0.02, depth_rel_ref=25.0)
    assert near != pytest.approx(far, rel=0.05)
    assert far < near        # 1 - beta*Z shrinks with depth


def test_effective_scale_invalid_alpha():
    assert np.isnan(cal.effective_scale(float("nan"), 0.0, 10.0))
    assert np.isnan(cal.effective_scale(-1.0, 0.0, 10.0))


def test_solve_affine_degenerate_input():
    out = cal.solve_affine(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2),
                           np.eye(3), np.array([0.5, 0, 0]), K)
    assert out["n"] == 0


# ── reprojection error ───────────────────────────────────────────────
# The cross-product residual used to solve for scale is proportional to the
# depths it is built from, so comparing it across candidate betas rewards
# shrinking depths rather than fitting better. These pin the pixel-space
# alternative that solve_affine actually scores with.

def test_reprojection_error_is_zero_for_the_true_fit():
    scale = 2.5
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array([0.5, 0, 0]), scale)
    err = cal.reprojection_error(pa, pb, z, R, t, K, alpha=1.0 / scale)
    assert err < 1e-9


def test_reprojection_error_grows_with_a_wrong_scale():
    scale = 2.5
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array([0.5, 0, 0]), scale)
    good = cal.reprojection_error(pa, pb, z, R, t, K, alpha=1.0 / scale)
    bad = cal.reprojection_error(pa, pb, z, R, t, K, alpha=1.0 / (scale * 1.5))
    assert bad > good
    assert bad > 0.5


def test_reprojection_error_does_not_reward_shrinking_depth():
    """The defect the pixel metric fixes: uniformly shrinking depth must not look
    like a better fit once the scale is re-solved for it."""
    scale = 2.5
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array([0.5, 0, 0]), scale)
    err_full, err_shrunk = [], []
    for factor in (1.0, 0.25):
        w = z * factor
        s, _ = cal.solve_scale(pa, pb, w, R, t, K)
        err = cal.reprojection_error(pa, pb, w, R, t, K, alpha=1.0 / s)
        (err_full if factor == 1.0 else err_shrunk).append(err)
    assert err_shrunk[0] == pytest.approx(err_full[0], abs=1e-6)


def test_affine_finds_zero_beta_on_pure_scale_data():
    """Regression: an unnormalized residual drove beta away from zero even when
    the depth was exactly proportional."""
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array([0.5, 0, 0]), 2.5, n=600)
    out = cal.solve_affine(pa, pb, z, R, t, K)
    assert abs(out["beta"]) < 1e-4
    assert out["alpha"] == pytest.approx(1 / 2.5, rel=1e-3)


def test_reprojection_error_degenerate():
    assert cal.reprojection_error(np.zeros((1, 2)), np.zeros((1, 2)), np.ones(1),
                                  np.eye(3), np.array([1.0, 0, 0]), K, 1.0) == float("inf")
    pa, pb, z, R, t = make_pair(np.zeros(3), np.array([0.5, 0, 0]), 2.0)
    assert cal.reprojection_error(pa, pb, z, R, t, K, float("nan")) == float("inf")
