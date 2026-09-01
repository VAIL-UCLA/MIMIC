"""Clip discovery for the stage scripts.

``--input assets/clips/*`` is the documented invocation, and the glob branch
has to behave like the directory branch: a match that is a clip folder holds
the stream one level down, and a match that is a plain file is only a clip if
it is a video. Keeping any file swept up the README shipped beside the clips
and reported it as the one clip found.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import _stage_common as sc


@pytest.fixture
def corpus(tmp_path):
    for uid in ("aaaa1111", "bbbb2222"):
        folder = tmp_path / uid
        folder.mkdir()
        (folder / "rgb_pinhole.mp4").write_bytes(b"")
        (folder / "route.mp4").write_bytes(b"")
        (folder / "poses_recorded.npy").write_bytes(b"")
    (tmp_path / "README.md").write_text("not a clip")
    return tmp_path


def test_glob_over_clip_folders_finds_the_stream(corpus):
    found = sc.find_clips(str(corpus / "*"))
    assert [p.name for p in found] == ["rgb_pinhole.mp4", "rgb_pinhole.mp4"]
    assert [p.parent.name for p in found] == ["aaaa1111", "bbbb2222"]


def test_glob_skips_non_video_files_beside_the_clips(corpus):
    found = sc.find_clips(str(corpus / "*"))
    assert not any(p.suffix == ".md" for p in found)


def test_glob_honours_the_stream_argument(corpus):
    found = sc.find_clips(str(corpus / "*"), stream="route.mp4")
    assert {p.name for p in found} == {"route.mp4"}


def test_directory_input_matches_glob_input(corpus):
    assert sc.find_clips(str(corpus)) == sc.find_clips(str(corpus / "*"))


def test_glob_directly_over_videos_still_works(corpus):
    found = sc.find_clips(str(corpus / "*" / "rgb_pinhole.mp4"))
    assert len(found) == 2


def test_a_named_file_is_taken_as_given(corpus):
    clip = corpus / "aaaa1111" / "rgb_pinhole.mp4"
    assert sc.find_clips(str(clip)) == [clip.resolve()]


def test_no_matches_is_empty(tmp_path):
    assert sc.find_clips(str(tmp_path / "nothing-here-*")) == []


# ---------------------------------------------------------------------------
# Where a rendered clip actually lands
#
# `--output x_act.mp4` does not produce that file; upstream treats the stem as
# a folder and writes x_act/gen.mp4 into it. A resume check that looks for the
# .mp4 never matches, so finished clips re-render and then fail on their own
# labels already existing.
# ---------------------------------------------------------------------------


def test_render_output_points_into_the_upstream_folder(tmp_path):
    import stage_2_action_augmentation as stage2

    target = tmp_path / "09294dbb_act.mp4"
    assert stage2.render_output(target) == tmp_path / "09294dbb_act" / "gen.mp4"


def test_render_output_detects_a_finished_clip(tmp_path):
    import stage_2_action_augmentation as stage2

    target = tmp_path / "09294dbb_act.mp4"
    assert not stage2.render_output(target).exists()
    folder = tmp_path / "09294dbb_act"
    folder.mkdir()
    (folder / "gen.mp4").write_bytes(b"")
    assert stage2.render_output(target).exists()
