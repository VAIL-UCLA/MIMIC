"""Window materialization.

The renderer only ever sees the leading TC_VIDEO_LENGTH * stride frames, so a
maneuver later in the recording has to be brought to the front by writing out
that stretch as its own clip bundle.
"""

import json

import numpy as np
import pytest

from mimic.action_augmentation import labels as lb
from mimic.action_augmentation import windows as win


def test_frame_span_counts_touched_frames_not_length_times_stride():
    # 49 samples at stride 4 touch frames 0, 4, ... 192 -> 193 frames.
    assert win.frame_span(49, 4) == 193
    assert win.frame_span(49, 1) == 49
    assert win.frame_span(1, 8) == 1


def test_window_start_is_zero_when_the_maneuver_already_fits():
    # 4 s maneuver at t=1 s ends at frame 100 of a 193-frame window.
    assert win.window_start_for(1.0, 4.0, 20.0, 49, 4, 400) == 0


def test_window_start_moves_to_the_maneuver_when_it_does_not_fit():
    # Ends at t=12.9 s = frame 258 > 193, so the window starts on the maneuver.
    start = win.window_start_for(8.9, 4.0, 20.0, 49, 4, 373)
    assert start == pytest.approx(178, abs=1)


def test_window_start_is_pulled_back_from_the_end_of_the_recording():
    start = win.window_start_for(17.0, 4.0, 20.0, 49, 4, 373)
    assert start + win.frame_span(49, 4) <= 373


def test_window_start_never_goes_negative_on_a_short_clip():
    assert win.window_start_for(9.0, 4.0, 20.0, 49, 4, 50) == 0


@pytest.fixture
def clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    folder = tmp_path / "src"
    folder.mkdir()
    video = folder / "rgb_pinhole.mp4"
    w, h, n = 64, 48, 60
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), i * 4 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    poses = np.zeros((n, 5), dtype=np.float64)
    poses[:, 0] = np.arange(n) * 0.1
    np.save(folder / "poses_recorded.npy", poses)
    (folder / "meta.json").write_text(json.dumps({"fps": 20.0}))
    return video


def test_materialize_writes_a_discoverable_bundle(clip, tmp_path):
    out = win.materialize(clip, tmp_path / "w", start=10, length=8, stride=2)
    assert out.is_file()
    # Sidecar and frame-rate discovery must work on the bundle unchanged.
    assert lb.find_sidecar(out).name == win.POSES_NAME
    assert lb.clip_fps(out) == 20.0


def test_materialize_slices_and_rebases_the_poses(clip, tmp_path):
    out = win.materialize(clip, tmp_path / "w", start=10, length=8, stride=2)
    data = lb.load_labels(lb.find_sidecar(out), fps=20.0)
    assert data.poses[0, 0] == pytest.approx(1.0)      # source frame 10
    assert data.times[0] == pytest.approx(0.0)         # re-based


def test_materialize_covers_the_whole_window(clip, tmp_path):
    out = win.materialize(clip, tmp_path / "w", start=10, length=8, stride=2)
    data = lb.load_labels(lb.find_sidecar(out), fps=20.0)
    assert len(data.poses) == win.frame_span(8, 2)


def test_materialize_records_its_provenance(clip, tmp_path):
    out = win.materialize(clip, tmp_path / "w", start=10, length=8, stride=2)
    meta = json.loads((out.parent / win.META_NAME).read_text())
    assert meta["window"] == {"start": 10, "length": 8, "stride": 2}
    assert meta["source_frames"][0] == 10


def test_materialize_rejects_an_offset_past_the_end(clip, tmp_path):
    with pytest.raises(ValueError):
        win.materialize(clip, tmp_path / "w", start=999, length=8, stride=2)
