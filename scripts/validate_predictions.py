"""Validate a predictions JSON.

    python scripts/validate_predictions.py --pred predictions_samples.json [--videos samples/]

Runs the repository's own schema checks (labels, 0 <= start < end <= duration,
no same-class overlap) and then, if the organisers' evaluate.py is present in
the repository root, the official check:
    python evaluate.py --pred <file> --validate-only
The exit code is non-zero if either check fails.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT, list_videos

from src.segments import validate_segments
from src.video import probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=str(ROOT / "predictions_samples.json"))
    ap.add_argument("--videos", nargs="*", default=[str(ROOT / "samples")], help="used to look up durations")
    args = ap.parse_args()
    try:
        with open(args.pred) as fh:
            preds = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot read {args.pred}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(preds, dict):
        print("predictions must be a JSON object {video: [[start, end, label], ...]}", file=sys.stderr)
        return 2
    durations = {v.name: probe(str(v)).duration for v in list_videos(args.videos)}
    failed = False
    for name, segs in sorted(preds.items()):
        dur = durations.get(name) or durations.get(Path(name).name)
        problems = validate_segments(segs, dur if dur else None)
        status = "OK" if not problems else "FAIL"
        print(f"[{status}] {name}: {len(segs) if isinstance(segs, list) else '?'} segments"
              + ("" if dur else " (duration unknown: end <= duration not checked)"))
        for p in problems:
            print(f"    - {p}")
        failed |= bool(problems)
    official = ROOT / "evaluate.py"
    if official.is_file():
        print(f"running official validator: python evaluate.py --pred {args.pred} --validate-only")
        rc = subprocess.run([sys.executable, str(official), "--pred", args.pred, "--validate-only"], cwd=ROOT).returncode
        failed |= rc != 0
    else:
        print("evaluate.py not found in the repository root - official validation skipped "
              "(copy the organisers' starter files into the repo root to enable it).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
