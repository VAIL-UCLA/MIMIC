"""Shared plumbing for the stage drivers.

The two stage scripts do the same shape of work — walk a corpus of clips, do
something slow to each one, and report as they go — so clip discovery, the
progress bar, and the run manifest live here.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

#: Video extensions considered when a directory is given instead of a glob.
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm")

#: Default stream to pick out of a clip folder. The pinhole stream is the
#: rectified one, and the one the depth model is meant to see.
DEFAULT_STREAM = "rgb_pinhole.mp4"


def find_clips(pattern: str, stream: str = DEFAULT_STREAM) -> list[Path]:
    """Resolve ``--input`` into a list of videos.

    Accepts a single file, a glob, or a directory. A directory is searched one
    level deep for ``stream`` first — the sample clips are laid out as
    ``<uid>/rgb_pinhole.mp4`` — and falls back to any video directly inside it.
    """
    import glob as globlib

    path = Path(pattern).expanduser()

    if path.is_file():
        return [path.resolve()]

    if path.is_dir():
        nested = sorted(path.glob(f"*/{stream}"))
        if nested:
            return [p.resolve() for p in nested]
        flat = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        )
        return [p.resolve() for p in flat]

    return [Path(p).resolve() for p in sorted(globlib.glob(str(path))) if Path(p).is_file()]


def clip_label(video: Path) -> str:
    """Short human name for a clip: its folder when the filename is generic."""
    if video.stem in ("rgb_pinhole", "rgb_fisheye", "video", "clip"):
        return f"{video.parent.name}/{video.stem}"
    return video.stem


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def banner(stage: str, title: str, fields: dict) -> None:
    """The block printed before work starts, so a long run is self-documenting."""
    print(f"\n\033[1m{stage} · {title}\033[0m")
    width = max((len(k) for k in fields), default=0)
    for key, value in fields.items():
        print(f"  {key:<{width}} : {value}")
    print()


#: Description column width. Fixed, so a changing description does not make
#: the bar jump around.
DESC_WIDTH = 52


def progress(items: list, desc: str, unit: str = "clip") -> tqdm:
    """A progress bar on stderr, and silence when stderr is redirected.

    The bar goes to stderr and the results to stdout, so piping a run into a log
    keeps the log free of half-drawn bar frames while the bar still shows on the
    terminal. ``disable=None`` drops it entirely when stderr is not a tty.
    """
    return tqdm(
        items,
        desc=desc.ljust(DESC_WIDTH)[:DESC_WIDTH],
        unit=unit,
        ncols=110,
        dynamic_ncols=False,
        bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        file=sys.stderr,
        leave=True,
        disable=None,
    )


def say(bar: tqdm | None, message: str) -> None:
    """Print a result line on stdout without fighting the bar for the cursor."""
    if bar is not None:
        bar.clear()
    print(message, flush=True)
    if bar is not None:
        bar.refresh()


@contextmanager
def step(bar: tqdm | None, message: str):
    """Show what is happening now, padded to a fixed width so the bar is steady."""
    if bar is not None:
        if len(message) > DESC_WIDTH:
            message = message[: DESC_WIDTH - 1] + "\u2026"
        bar.set_description_str(message.ljust(DESC_WIDTH))
    yield


def write_manifest(path: Path, stage: str, config: dict, records: list[dict]) -> Path:
    """Record what a run produced, so a corpus can be audited after the fact."""
    written = sum(1 for r in records if r.get("status") == "written")
    skipped = sum(1 for r in records if r.get("status") == "skipped")
    failed = sum(1 for r in records if r.get("status") == "failed")
    payload = {
        "stage": stage,
        "config": config,
        "summary": {
            "total": len(records),
            "written": written,
            "skipped": skipped,
            "failed": failed,
        },
        "items": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def summarize(stage: str, records: list[dict], elapsed: float, manifest: Path | None) -> int:
    """Closing report. Returns the process exit code."""
    written = sum(1 for r in records if r.get("status") == "written")
    skipped = sum(1 for r in records if r.get("status") == "skipped")
    failed = [r for r in records if r.get("status") == "failed"]

    print(f"\n\033[1m{stage} complete\033[0m — {written} written, {skipped} skipped, "
          f"{len(failed)} failed  ({human_time(elapsed)})")
    for record in failed:
        print(f"  ! {record.get('clip', '?')}: {record.get('error', 'unknown error')}")
    if manifest is not None:
        print(f"  manifest: {manifest}")
    return 1 if failed and written == 0 else 0
