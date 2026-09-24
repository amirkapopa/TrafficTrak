"""Make the repository root importable when a script is run as `python scripts/<name>.py`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def list_videos(paths) -> list[Path]:
    """Expand files / directories into a sorted list of video files."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += [q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXTS]
        elif p.is_file():
            out.append(p)
    return sorted(set(out))
