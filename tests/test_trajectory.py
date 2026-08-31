"""Tests for the corrective-behavior trajectory math."""

import numpy as np
import pytest

from mimic.action_augmentation import trajectory as tj


def straight_path(n=81, fps=20.0, speed=1.0):
    """Straight path along +x at constant speed."""
    times = np.arange(n) / fps
    poses = np.zeros((n, 3))
    poses[:, 0] = times * speed
    return poses, times


def curved_path(n=81, fps=20.0, speed=1.0, radius=10.0):
    """Constant-curvature arc, to check the offset follows the normal."""
    times = np.arange(n) / fps
    arc = times * speed / radius
    poses = np.stack([radius * np.sin(arc), radius * (1 - np.cos(arc)), arc], axis=-1)
    return poses, times


# ── offset profile ───────────────────────────────────────────────────

@pytest.mark.parametrize("profile", tj.PROFILES)
def test_profile_endpoints_and_peak(profile):
    """0 at both ends, peak == strength at the midpoint."""
    h, s = 4.0, 0.5
    assert tj.lateral_offset_profile(0.0, s, h, profile) == pytest.approx(0.0, abs=1e-12)
    assert tj.lateral_offset_profile(h, s, h, profile) == pytest.approx(0.0, abs=1e-12)
    assert tj.lateral_offset_profile(h / 2, s, h, profile) == pytest.approx(s)


@pytest.mark.parametrize("profile", tj.PROFILES)
def test_profile_stays_within_strength(profile):
    t = np.linspace(0, 4, 400)
    d = tj.lateral_offset_profile(t, 0.5, 4.0, profile)
    assert d.min() >= -1e-12
    assert d.max() <= 0.5 + 1e-12


def test_profile_zero_outside_window():
    t = np.array([-1.0, -0.01, 4.01, 9.0])
    assert np.allclose(tj.lateral_offset_profile(t, 0.5, 4.0), 0.0)


def test_raised_cosine_has_zero_velocity_at_ends():
    """Smooth entry and exit — no lateral velocity step at the boundaries."""
    t = np.linspace(0, 4, 4001)
    d = tj.lateral_offset_profile(t, 0.5, 4.0, "raised_cosine")
    v = np.gradient(d, t)
    assert abs(v[0]) < 1e-3
    assert abs(v[-1]) < 1e-3


def test_triangle_velocity_is_discontinuous_at_peak():
    """Contrast case: the baseline profile does step, which is why it is not default."""
    t = np.linspace(0, 4, 4001)
    v = np.gradient(tj.lateral_offset_profile(t, 0.5, 4.0, "triangle"), t)
    mid = len(t) // 2
    assert abs(v[mid - 10] - v[mid + 10]) > 0.1


def test_negative_strength_goes_right():
    assert tj.lateral_offset_profile(2.0, -0.5, 4.0) == pytest.approx(-0.5)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="Unknown profile"):
        tj.lateral_offset_profile(1.0, 0.5, 4.0, "bogus")


# ── SE(2) ────────────────────────────────────────────────────────────

def test_se2_inverse_roundtrip():
    rng = np.random.default_rng(0)
    poses = rng.normal(size=(50, 3))
    poses[:, 2] = tj.wrap_angle(poses[:, 2] * 3)
    for p in poses:
        m = tj.se2_matrix(p) @ tj.se2_matrix(tj.se2_inverse(p))
        assert np.allclose(m, np.eye(3), atol=1e-12)


def test_relative_pose_of_self_is_identity():
    rng = np.random.default_rng(1)
    p = rng.normal(size=3)
    assert np.allclose(tj.relative_pose(p, p), 0.0, atol=1e-12)


def test_relative_pose_roundtrip():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=3)
    tgt = rng.normal(size=(20, 3))
    back = tj._to_global(ref, tj.relative_pose(ref, tgt))
    assert np.allclose(back[:, :2], tgt[:, :2], atol=1e-12)
    assert np.allclose(tj.wrap_angle(back[:, 2] - tgt[:, 2]), 0.0, atol=1e-12)


def test_relative_pose_known_case():
    """Target 1 m ahead, observer rotated 90 deg left → target is 1 m to the right."""
    ref = np.array([0.0, 0.0, np.pi / 2])
    out = tj.relative_pose(ref, np.array([1.0, 0.0, 0.0]))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(-1.0)


def test_wrap_angle():
    # +pi and -pi are the same heading; wrap_angle uses the [-pi, pi) branch.
    assert abs(tj.wrap_angle(3 * np.pi)) == pytest.approx(np.pi)
    assert tj.wrap_angle(0.5) == pytest.approx(0.5)
    assert tj.wrap_angle(2 * np.pi + 0.5) == pytest.approx(0.5)
    assert tj.wrap_angle(-2 * np.pi - 0.5) == pytest.approx(-0.5)


# ── lateral offset ───────────────────────────────────────────────────

def test_offset_on_straight_path_moves_along_y():
    poses, _ = straight_path()
    off = np.full(len(poses), 0.5)
    out = tj.apply_lateral_offset(poses, off, recompute_yaw=False)
    assert np.allclose(out[:, 1], 0.5)
    assert np.allclose(out[:, 0], poses[:, 0])


def test_offset_is_perpendicular_on_curved_path():
    """Displaced point must sit exactly `offset` from the original, along the normal."""
    poses, _ = curved_path()
    off = np.full(len(poses), 0.5)
    out = tj.apply_lateral_offset(poses, off, recompute_yaw=False)
    delta = out[:, :2] - poses[:, :2]
    assert np.allclose(np.hypot(delta[:, 0], delta[:, 1]), 0.5)
    tangent = np.stack([np.cos(poses[:, 2]), np.sin(poses[:, 2])], axis=-1)
    assert np.allclose(np.einsum("ij,ij->i", delta, tangent), 0.0, atol=1e-12)


def test_yaw_follows_tangent_not_original():
    """The whole point of recompute_yaw: heading turns into the maneuver."""
    poses, times = straight_path()
    new, off = tj.deviate_and_recover(poses, times, strength=0.5)
    assert np.all(off >= -1e-12)
    # Drifting left in the first phase → heading points left (positive yaw).
    early = (times > 0.5) & (times < 1.5)
    assert np.all(new[early, 2] > 0.01)
    # Rejoining in the second phase → heading points back right.
    late = (times > 2.5) & (times < 3.5)
    assert np.all(new[late, 2] < -0.01)


def test_zero_strength_is_identity():
    poses, times = straight_path()
    new, off = tj.deviate_and_recover(poses, times, strength=0.0)
    assert np.allclose(off, 0.0)
    assert np.allclose(new[:, :2], poses[:, :2], atol=1e-12)


def test_offset_shape_validated():
    poses, _ = straight_path(n=10)
    with pytest.raises(ValueError, match="offsets must be"):
        tj.apply_lateral_offset(poses, np.zeros(5))
    with pytest.raises(ValueError, match=r"poses must be"):
        tj.apply_lateral_offset(np.zeros((10, 2)), np.zeros(10))


def test_stationary_path_keeps_recorded_heading():
    """A parked robot has no tangent; heading must not blow up."""
    poses = np.zeros((20, 3))
    poses[:, 2] = 0.3
    out = tj.apply_lateral_offset(poses, np.zeros(20))
    assert np.allclose(out[:, 2], 0.3)


# ── labels ───────────────────────────────────────────────────────────

def test_label_shape_and_first_waypoint_is_ahead():
    poses, times = straight_path()
    wp = tj.waypoints_from_path(poses, times)
    assert wp.shape == (len(poses), len(tj.DEFAULT_LABEL_TIMES), 3)
    # Straight path at 1 m/s: waypoint at 0.2 s sits 0.2 m ahead, no lateral.
    assert wp[0, 0, 0] == pytest.approx(0.2, abs=1e-9)
    assert wp[0, 0, 1] == pytest.approx(0.0, abs=1e-9)


def test_label_is_expressed_in_ego_frame():
    """Same maneuver at different points on the path yields the same local label."""
    poses, times = straight_path(n=200)
    wp = tj.waypoints_from_path(poses, times)
    assert np.allclose(wp[10], wp[50], atol=1e-9)


def test_label_on_curved_path_bends():
    poses, times = curved_path(n=200)
    wp = tj.waypoints_from_path(poses, times)
    # Arc turns left, so far waypoints have positive lateral offset.
    assert wp[0, -1, 1] > 0.05


def test_tail_can_be_nan_instead_of_clamped():
    poses, times = straight_path(n=40, fps=20.0)   # 2 s of clip, 5 s horizon
    clamped = tj.waypoints_from_path(poses, times, clamp_tail=True)
    nan = tj.waypoints_from_path(poses, times, clamp_tail=False)
    assert np.isfinite(clamped).all()
    assert np.isnan(nan).any()


def test_label_times_mismatch_rejected():
    poses, times = straight_path(n=10)
    with pytest.raises(ValueError, match="must match"):
        tj.waypoints_from_path(poses, times[:5])


# ── end-to-end augmentation ──────────────────────────────────────────

def test_deviate_recover_label_returns_to_path():
    """The maneuver must actually come back: offset ends where it started."""
    poses, times = straight_path(n=161, fps=20.0)  # 8 s
    out = tj.build_augmented_label(poses, times, strength=0.5, horizon=4.0)
    assert out["offsets"][0] == pytest.approx(0.0, abs=1e-12)    # t = 0 s, on path
    assert out["offsets"][40] == pytest.approx(0.5)              # t = 2 s, peak
    assert out["offsets"][80] == pytest.approx(0.0, abs=1e-12)   # t = 4 s, rejoined
    assert np.allclose(out["offsets"][80:], 0.0, atol=1e-12)     # and stays there


def test_deviate_recover_peak_matches_strength():
    poses, times = straight_path(n=161, fps=20.0)
    for s in (0.2, 0.5, 1.0, -0.75):
        out = tj.build_augmented_label(poses, times, strength=s, horizon=4.0)
        assert out["offsets"][np.argmax(np.abs(out["offsets"]))] == pytest.approx(s)


def test_reexpress_label_points_back_toward_recorded_path():
    """Offset left by 0.5 m → the recorded path is 0.5 m to the right."""
    poses, times = straight_path(n=200)
    out = tj.build_augmented_label(poses, times, strength=0.5, mode="reexpress")
    assert np.allclose(out["offsets"], 0.5)
    assert out["waypoints"][0, 0, 1] == pytest.approx(-0.5, abs=1e-6)


def test_reexpress_and_deviate_differ():
    poses, times = straight_path(n=161, fps=20.0)
    a = tj.build_augmented_label(poses, times, strength=0.5, mode="deviate_recover")
    b = tj.build_augmented_label(poses, times, strength=0.5, mode="reexpress")
    assert not np.allclose(a["waypoints"], b["waypoints"])


def test_unknown_mode_rejected():
    poses, times = straight_path()
    with pytest.raises(ValueError, match="Unknown mode"):
        tj.build_augmented_label(poses, times, 0.5, mode="bogus")


def test_start_time_shifts_the_maneuver():
    poses, times = straight_path(n=201, fps=20.0)
    out = tj.build_augmented_label(poses, times, 0.5, horizon=4.0, start_time=2.0)
    assert out["offsets"][40] == pytest.approx(0.0, abs=1e-12)   # t=2 s, start
    assert out["offsets"][80] == pytest.approx(0.5)              # t=4 s, peak


def _unsaturated(times):
    """Frames whose full label horizon fits inside the clip."""
    return times + tj.DEFAULT_LABEL_TIMES[-1] <= times[-1]


def test_poses_from_waypoints_recovers_a_straight_path():
    """Round-trip: path -> ego-frame labels -> path, over the unsaturated region."""
    poses, times = straight_path(n=300, fps=20.0, speed=1.0)   # 15 s, 5 s horizon
    wp = tj.waypoints_from_path(poses, times)
    rebuilt = tj.poses_from_waypoints(wp, times)
    ok = _unsaturated(times)
    assert ok.sum() > 100, "need a decent unsaturated region to test against"
    assert np.allclose(rebuilt[ok, 0], poses[ok, 0], atol=1e-6)
    assert np.allclose(rebuilt[ok, 1], 0.0, atol=1e-6)


def test_poses_from_waypoints_recovers_a_curved_path():
    """Chaining chord steps drifts on curves. Bound it rather than pretend it is
    exact: ~7 cm over 10 s on a 10 m arc at 1 m/s. Use a poses sidecar for exact
    work — this path is the label-only fallback."""
    poses, times = curved_path(n=300, fps=20.0, speed=1.0, radius=10.0)
    rebuilt = tj.poses_from_waypoints(tj.waypoints_from_path(poses, times), times)
    ok = _unsaturated(times)
    err = np.linalg.norm(rebuilt[ok, :2] - poses[ok, :2], axis=1)
    assert err.max() < 0.10
    # Drift accumulates monotonically rather than being random noise.
    assert err[-1] > err[len(err) // 2]


def test_clamped_tail_under_reconstructs():
    """Documents the limit: past the clip end the label saturates, so the
    reconstructed path falls short. Callers should drop or ignore that tail."""
    poses, times = straight_path(n=300, fps=20.0, speed=1.0)
    rebuilt = tj.poses_from_waypoints(tj.waypoints_from_path(poses, times), times)
    assert rebuilt[-1, 0] < poses[-1, 0] - 0.01


def test_endpoint_yaw_is_not_biased():
    """Array endpoints use a one-sided difference. At first order that biases the
    heading by half a step — ~4 deg at t=0, where the maneuver has not started
    and yaw should be zero. Second-order differencing removes it."""
    poses, times = straight_path(n=161, fps=20.0)
    new, _ = tj.deviate_and_recover(poses, times, strength=0.5, horizon=4.0)
    assert abs(np.degrees(new[0, 2])) < 0.01
    assert abs(np.degrees(new[-1, 2])) < 0.01


def test_yaw_at_maneuver_end_is_small():
    """t = 4 s is interior, so it keeps the central difference. The profile has
    zero slope there but non-zero curvature, leaving an O(h^2) residual — well
    under a degree, and not the endpoint bias above."""
    poses, times = straight_path(n=161, fps=20.0)
    new, _ = tj.deviate_and_recover(poses, times, strength=0.5, horizon=4.0)
    assert abs(np.degrees(new[80, 2])) < 1.0
    assert np.allclose(new[81:, 2], 0.0, atol=1e-9)   # fully back on path after


def test_peak_yaw_matches_lateral_velocity():
    """Closed form: yaw peaks at atan(max lateral speed / forward speed)."""
    speed = 1.0
    poses, times = straight_path(n=161, fps=20.0, speed=speed)
    new, _ = tj.deviate_and_recover(poses, times, strength=0.5, horizon=4.0)
    # d(t) = s/2 (1 - cos 2*pi*t/H)  ->  max |d'| = s*pi/H
    expected = np.arctan2(0.5 * np.pi / 4.0, speed)
    assert np.abs(new[:, 2]).max() == pytest.approx(expected, rel=1e-3)


# =====================================================================
# Recorded footage is not a clean straight run: robots park at crossings,
# reverse, and occasionally record a corrupt heading. Each of these broke
# the maneuver in a different way.
# =====================================================================


def _straight_path(n=100, fps=20.0, speed=1.5):
    t = np.arange(n) / fps
    x = speed * t
    return np.stack([x, np.zeros(n), np.zeros(n)], axis=1), t


def test_parked_robot_gets_no_manufactured_heading():
    """Displacing a stationary robot sideways gives a tangent perpendicular to
    the way it faces. The heading must stay put instead."""
    n, fps = 100, 20.0
    t = np.arange(n) / fps
    poses = np.zeros((n, 3))
    poses[:, 2] = 0.3            # parked, facing a fixed direction
    offsets = tj.lateral_offset_profile(t, 0.5, 4.0, "raised_cosine")
    out = tj.apply_lateral_offset(poses, offsets)
    assert np.allclose(out[:, 2], 0.3, atol=1e-9)


def test_parked_robot_still_gets_displaced():
    """Only the heading is held; the offset itself still applies."""
    n = 100
    poses = np.zeros((n, 3))
    offsets = np.full(n, 0.4)
    out = tj.apply_lateral_offset(poses, offsets)
    np.testing.assert_allclose(out[:, 1], 0.4)


def test_reversing_robot_keeps_its_heading():
    """A robot backing up has a path tangent 180 degrees from where it faces.
    Taking the tangent as the heading flips the whole maneuver."""
    n, fps = 100, 20.0
    t = np.arange(n) / fps
    poses = np.stack([-1.5 * t, np.zeros(n), np.zeros(n)], axis=1)  # facing +x, moving -x
    offsets = tj.lateral_offset_profile(t, 0.5, 4.0, "raised_cosine")
    out = tj.apply_lateral_offset(poses, offsets)
    delta = np.degrees(np.abs(tj.wrap_angle(out[:, 2] - poses[:, 2])))
    assert delta.max() < 30.0, f"reversing produced {delta.max():.1f} deg of spurious yaw"


def test_forward_and_reverse_yaw_deltas_agree():
    """The maneuver's turn should not depend on the sign of travel."""
    n, fps = 100, 20.0
    t = np.arange(n) / fps
    offsets = tj.lateral_offset_profile(t, 0.5, 4.0, "raised_cosine")

    fwd = np.stack([1.5 * t, np.zeros(n), np.zeros(n)], axis=1)
    rev = np.stack([-1.5 * t, np.zeros(n), np.zeros(n)], axis=1)
    d_fwd = tj.wrap_angle(tj.apply_lateral_offset(fwd, offsets)[:, 2] - fwd[:, 2])
    d_rev = tj.wrap_angle(tj.apply_lateral_offset(rev, offsets)[:, 2] - rev[:, 2])
    np.testing.assert_allclose(d_fwd, -d_rev, atol=1e-6)


def test_path_speed_matches_a_constant_rate():
    poses, t = _straight_path(speed=1.5)
    speed = tj.path_speed(poses, t)
    np.testing.assert_allclose(speed, 1.5, rtol=1e-9)


def test_yaw_consistency_catches_a_flipped_sample():
    """The failure seen in the sample data: one heading 180 degrees out at a
    wrap boundary, everything else clean."""
    poses, t = _straight_path()
    poses[40, 2] = np.pi
    ok = tj.yaw_is_consistent(poses, t)
    assert not ok[40]
    assert ok.sum() == len(poses) - 1


def test_yaw_consistency_ignores_stationary_samples():
    """A parked robot has no direction of travel to disagree with."""
    n = 60
    t = np.arange(n) / 20.0
    poses = np.zeros((n, 3))
    poses[:, 2] = 2.0
    assert tj.yaw_is_consistent(poses, t).all()


def test_maneuver_start_skips_a_parked_opening():
    """Half parked, then moving: the maneuver belongs in the moving half."""
    fps, n = 20.0, 200
    t = np.arange(n) / fps
    x = np.where(t < 5.0, 0.0, 1.5 * (t - 5.0))
    poses = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    start, fraction = tj.best_maneuver_start(poses, t, 4.0)
    assert start >= 5.0
    assert fraction == pytest.approx(1.0)


def test_maneuver_start_avoids_a_corrupt_heading():
    poses, t = _straight_path(n=400)
    poses[100, 2] = np.pi
    start, fraction = tj.best_maneuver_start(poses, t, 4.0)
    window = (t >= start) & (t <= start + 4.0)
    assert not window[100], "the maneuver window still contains the bad sample"
    assert fraction == pytest.approx(1.0)


def test_maneuver_start_reports_a_hopeless_clip():
    """Parked throughout: there is nowhere good, and the caller must be told."""
    n = 200
    t = np.arange(n) / 20.0
    poses = np.zeros((n, 3))
    start, fraction = tj.best_maneuver_start(poses, t, 4.0)
    assert fraction == pytest.approx(0.0)
    assert start == pytest.approx(0.0)


def test_maneuver_start_stays_within_the_clip():
    poses, t = _straight_path(n=200)
    start, _ = tj.best_maneuver_start(poses, t, 4.0)
    assert start + 4.0 <= t[-1] + 1e-9


# =====================================================================
# Sizing the offset against speed
# =====================================================================


def test_peak_yaw_closed_form():
    assert tj.peak_yaw(0.5, 4.0, 1.0) == pytest.approx(np.arctan2(0.5 * np.pi / 4.0, 1.0))


def test_peak_yaw_of_a_stopped_robot_is_a_right_angle():
    """No forward motion, so any sideways move is a 90 degree turn."""
    assert tj.peak_yaw(0.5, 4.0, 0.0) == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("speed", [0.3, 1.0, 1.7, 3.0])
def test_speed_relative_offset_holds_the_yaw_constant(speed):
    """The whole point: the same demand on the robot at any speed."""
    scale, horizon = 0.34, 4.0
    strength = tj.strength_for_speed(scale, speed)
    expected = np.arctan(scale * np.pi / horizon)
    assert tj.peak_yaw(strength, horizon, speed) == pytest.approx(expected)


def test_a_fixed_offset_does_not_hold_the_yaw_constant():
    """The behaviour being replaced: a slow robot is asked to swerve far harder."""
    slow = tj.peak_yaw(0.6, 4.0, 0.3)
    fast = tj.peak_yaw(0.6, 4.0, 3.0)
    assert np.degrees(slow) > 55.0
    assert np.degrees(fast) < 10.0


def test_speed_relative_offset_scales_with_speed():
    assert tj.strength_for_speed(0.34, 2.0) == pytest.approx(2 * tj.strength_for_speed(0.34, 1.0))


def test_speed_relative_offset_is_never_negative():
    assert tj.strength_for_speed(-0.34, 2.0) >= 0.0
    assert tj.strength_for_speed(0.34, -2.0) == 0.0


def test_speed_relative_offset_measured_on_a_real_maneuver():
    """End to end: build the path and read the yaw back off it."""
    scale, horizon, speed = 0.34, 4.0, 2.0
    poses, times = straight_path(n=161, fps=20.0, speed=speed)
    strength = tj.strength_for_speed(scale, speed)
    new, _ = tj.deviate_and_recover(poses, times, strength=strength, horizon=horizon)
    assert np.abs(new[:, 2]).max() == pytest.approx(np.arctan(scale * np.pi / horizon), rel=2e-3)
