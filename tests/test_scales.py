"""Per-clip scale sidecars: naming, round-trip, and how a scale is resolved."""

import json

import pytest

from mimic.action_augmentation import scales as sc


def test_sidecar_sits_beside_the_clip(tmp_path):
    video = tmp_path / "rgb_pinhole.mp4"
    assert sc.scale_path(video) == tmp_path / "rgb_pinhole.scale.json"


def test_sidecar_does_not_collide_with_a_json_label(tmp_path):
    """A clip may carry both clip.json labels and clip.scale.json."""
    video = tmp_path / "clip.mp4"
    assert sc.scale_path(video).name != "clip.json"


def test_round_trip(tmp_path):
    video = tmp_path / "clip.mp4"
    window = sc.make_window(49, 1)
    written = sc.save_scale(video, 2.5012, window, labels=tmp_path / "clip.npy",
                            calibration={"n_pairs": 39})
    assert written == sc.scale_path(video)

    record = sc.load_scale(video)
    assert record["scale"] == pytest.approx(2.5012)
    assert record["window"] == window
    assert record["clip"] == "clip.mp4"
    assert record["labels"] == "clip.npy"
    assert record["calibration"]["n_pairs"] == 39
    assert record["version"] == sc.SCHEMA_VERSION


def test_window_records_the_leading_frames():
    """The renderer only ever consumes the head of a clip, so start is 0."""
    assert sc.make_window(49, 2) == {"start": 0, "length": 49, "stride": 2}


def test_load_returns_none_when_absent(tmp_path):
    assert sc.load_scale(tmp_path / "clip.mp4") is None


def test_load_rejects_a_newer_schema(tmp_path):
    video = tmp_path / "clip.mp4"
    path = sc.scale_path(video)
    path.write_text(json.dumps({"version": sc.SCHEMA_VERSION + 1, "scale": 1.0}))
    with pytest.raises(ValueError, match="schema version"):
        sc.load_scale(video)


def test_load_rejects_a_record_without_a_scale(tmp_path):
    video = tmp_path / "clip.mp4"
    sc.scale_path(video).write_text(json.dumps({"version": 1, "clip": "clip.mp4"}))
    with pytest.raises(ValueError, match="no 'scale' field"):
        sc.load_scale(video)


def test_explicit_scale_wins_over_the_sidecar(tmp_path):
    video = tmp_path / "clip.mp4"
    sc.save_scale(video, 2.5, sc.make_window(49))
    value, provenance = sc.resolve_scale(video, "9.0")
    assert value == pytest.approx(9.0)
    assert "command line" in provenance


def test_auto_reads_the_sidecar(tmp_path):
    video = tmp_path / "clip.mp4"
    sc.save_scale(video, 2.5, sc.make_window(49))
    value, provenance = sc.resolve_scale(video, sc.AUTO, window=sc.make_window(49))
    assert value == pytest.approx(2.5)
    assert provenance == "clip.scale.json"
    assert "WARNING" not in provenance


def test_auto_without_a_sidecar_points_at_the_calibrator(tmp_path):
    video = tmp_path / "clip.mp4"
    with pytest.raises(sc.ScaleNotFound, match="calibrate_clips"):
        sc.resolve_scale(video, sc.AUTO)


def test_each_clip_resolves_to_its_own_scale(tmp_path):
    """The whole point: two clips in one tree keep independent scales."""
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    sc.save_scale(a, 2.5, sc.make_window(49))
    sc.save_scale(b, 0.7, sc.make_window(49))
    assert sc.resolve_scale(a, sc.AUTO)[0] == pytest.approx(2.5)
    assert sc.resolve_scale(b, sc.AUTO)[0] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "rendering, expected",
    [
        (sc.make_window(49, 1), None),
        (sc.make_window(24, 1), "length"),
        (sc.make_window(49, 2), "stride"),
    ],
)
def test_window_mismatch_is_detected(rendering, expected):
    record = {"window": sc.make_window(49, 1)}
    result = sc.window_mismatch(record, rendering)
    if expected is None:
        assert result is None
    else:
        assert expected in result


def test_mismatched_window_warns_in_the_provenance(tmp_path):
    """A scale calibrated on other frames must not be applied silently:
    DepthCrafter normalizes over the window, so the units differ."""
    video = tmp_path / "clip.mp4"
    sc.save_scale(video, 2.5, sc.make_window(40, 1))
    value, provenance = sc.resolve_scale(video, sc.AUTO, window=sc.make_window(49, 1))
    assert value == pytest.approx(2.5)
    assert "WARNING" in provenance and "length" in provenance


def test_a_record_without_a_window_is_flagged():
    assert sc.window_mismatch({}, sc.make_window(49)) == "the sidecar records no window"


@pytest.mark.parametrize("bad", ["abc", "", "-1", "0"])
def test_bad_explicit_scales_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        sc.resolve_scale(tmp_path / "clip.mp4", bad)


def test_scale_file_overrides_the_location(tmp_path):
    video = tmp_path / "clip.mp4"
    elsewhere = tmp_path / "store" / "custom.json"
    sc.save_scale(video, 3.3, sc.make_window(49), path=elsewhere)
    assert not sc.scale_path(video).exists()
    assert sc.resolve_scale(video, sc.AUTO, path=elsewhere)[0] == pytest.approx(3.3)
