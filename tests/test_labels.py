"""Sidecar loading: discovery, the three formats, and the positional .npy rules."""

import json

import numpy as np
import pytest

from mimic.action_augmentation import labels as lb


def _poses(n=10):
    t = np.linspace(0.0, 1.0, n)
    return np.stack([t * 2.0, t * 0.5, t * 0.1], axis=1)


def test_finds_sidecar_beside_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    sidecar = tmp_path / "clip.npy"
    np.save(sidecar, _poses())
    assert lb.find_sidecar(video) == sidecar


def test_missing_sidecar_lists_what_it_tried(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    with pytest.raises(FileNotFoundError, match=r"clip\.npy.*clip\.npz.*clip\.json"):
        lb.find_sidecar(video)


def test_bare_npy_read_as_poses(tmp_path):
    path = tmp_path / "clip.npy"
    poses = _poses()
    np.save(path, poses)
    data = lb.load_labels(path, fps=20.0)
    np.testing.assert_allclose(data.poses, poses)
    np.testing.assert_allclose(data.times, np.arange(len(poses)) / 20.0)
    assert not data.reconstructed


def test_bare_npy_with_trailing_columns_keeps_only_the_pose(tmp_path):
    """The sample clips store (x, y, yaw, v, w); v and w are derived, not pose."""
    path = tmp_path / "clip.npy"
    poses = _poses()
    extra = np.concatenate([poses, np.full((len(poses), 2), 7.0)], axis=1)
    assert extra.shape[1] == 5
    np.save(path, extra)
    data = lb.load_labels(path)
    np.testing.assert_allclose(data.poses, poses)


def test_bare_npy_too_few_columns_is_rejected(tmp_path):
    path = tmp_path / "clip.npy"
    np.save(path, np.zeros((10, 2)))
    with pytest.raises(ValueError, match=r"\(N, >=3\) poses"):
        lb.load_labels(path)


def test_bare_npy_read_as_waypoints(tmp_path):
    path = tmp_path / "clip.npy"
    waypoints = np.zeros((6, 4, 3))
    waypoints[:, :, 0] = np.arange(1, 5)  # each frame looks 1..4 m ahead
    np.save(path, waypoints)
    data = lb.load_labels(path)
    assert data.reconstructed
    assert data.poses.shape == (6, 3)


def test_npz_and_json_agree_with_npy(tmp_path):
    poses = _poses()
    times = np.arange(len(poses)) / 20.0

    npz = tmp_path / "a.npz"
    np.savez(npz, poses=poses, times=times)
    js = tmp_path / "a.json"
    js.write_text(json.dumps({"poses": poses.tolist(), "times": times.tolist()}))

    from_npz = lb.load_labels(npz)
    from_json = lb.load_labels(js)
    np.testing.assert_allclose(from_npz.poses, poses)
    np.testing.assert_allclose(from_json.poses, poses)
    np.testing.assert_allclose(from_npz.times, times)
    np.testing.assert_allclose(from_json.times, times)


def test_stored_times_override_fps(tmp_path):
    poses = _poses(5)
    times = np.array([0.0, 0.3, 0.9, 1.1, 2.0])
    path = tmp_path / "a.npz"
    np.savez(path, poses=poses, times=times)
    data = lb.load_labels(path, fps=1000.0)
    np.testing.assert_allclose(data.times, times)
    assert data.duration == pytest.approx(2.0)


def test_times_length_mismatch_is_rejected(tmp_path):
    path = tmp_path / "a.npz"
    np.savez(path, poses=_poses(5), times=np.zeros(4))
    with pytest.raises(ValueError, match="4 entries but there are 5"):
        lb.load_labels(path)


def test_finds_a_pose_file_named_after_its_contents(tmp_path):
    """Recorded clips keep several streams beside one pose file:
    <uid>/rgb_pinhole.mp4 with <uid>/poses_recorded.npy."""
    video = tmp_path / "rgb_pinhole.mp4"
    video.touch()
    sidecar = tmp_path / "poses_recorded.npy"
    np.save(sidecar, _poses())
    assert lb.find_sidecar(video) == sidecar


def test_a_sidecar_named_after_the_video_wins(tmp_path):
    video = tmp_path / "rgb_pinhole.mp4"
    video.touch()
    np.save(tmp_path / "poses_recorded.npy", _poses())
    own = tmp_path / "rgb_pinhole.npy"
    np.save(own, _poses())
    assert lb.find_sidecar(video) == own


def test_explicit_labels_beat_both(tmp_path):
    video = tmp_path / "rgb_pinhole.mp4"
    video.touch()
    np.save(tmp_path / "poses_recorded.npy", _poses())
    chosen = tmp_path / "elsewhere.npy"
    np.save(chosen, _poses())
    assert lb.find_sidecar(video, explicit=chosen) == chosen


def test_missing_sidecar_names_the_folder_it_searched(tmp_path):
    video = tmp_path / "rgb_pinhole.mp4"
    video.touch()
    with pytest.raises(FileNotFoundError, match="poses_recorded.npy"):
        lb.find_sidecar(video)


# --- offset sizing ---------------------------------------------------


def test_sample_strength_absolute_wins():
    from mimic.action_augmentation import augment_action as aa

    assert aa.sample_strength(1, strength=0.42) == pytest.approx(0.42)


def test_sample_strength_from_speed():
    from mimic.action_augmentation import augment_action as aa

    value = aa.sample_strength(1, strength_scale=0.34, speed=2.0)
    assert abs(value) == pytest.approx(0.68)


def test_sample_strength_from_speed_needs_a_speed():
    from mimic.action_augmentation import augment_action as aa

    with pytest.raises(ValueError, match="needs the clip's speed"):
        aa.sample_strength(1, strength_scale=0.34)


def test_sample_strength_needs_some_sizing():
    from mimic.action_augmentation import augment_action as aa

    with pytest.raises(ValueError, match="strength_scale or strength_range"):
        aa.sample_strength(1)


def test_sample_strength_takes_both_sides_over_a_corpus():
    from mimic.action_augmentation import augment_action as aa

    signs = [np.sign(aa.sample_strength(s, strength_range=(0.3, 0.8))) for s in range(400)]
    assert 0.4 < signs.count(1.0) / len(signs) < 0.6
