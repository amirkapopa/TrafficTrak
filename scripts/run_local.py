"""Development runner with visual outputs (the official run is run_submission.py).

    python scripts/run_local.py --videos samples/ --risk --visualize
      -> outputs/vis/<video>_annotated.mp4, _timeline.png, _events.json,
         outputs/risk/<video>.csv, outputs/run_report.json (timings)

The predictions file uses the official format, so it can be scored directly:
    {"team": ..., "videos": {"clip.mp4": {"events": [[s, e, label], ...], "risk": [[t, p], ...]}}}
    python evaluate.py --pred <out> --gt labels/dev_labels.json --per-video
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from _bootstrap import ROOT, list_videos

import solution
from src.pipeline import analyze_video
from src.segments import validate_segments
from src.video import probe
from src.visualize import plot_timeline, render_annotated_video


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", default=[str(ROOT / "samples")])
    ap.add_argument("--out", default=str(ROOT / "predictions_samples.json"))
    ap.add_argument("--risk", action="store_true", help="also run the causal RiskEstimator and save per-frame CSVs")
    ap.add_argument("--visualize", action="store_true", help="annotated video + timeline/risk PNGs")
    ap.add_argument("--outdir", default=str(ROOT / "outputs"))
    ap.add_argument("--team", default="TrafficTrak")
    args = ap.parse_args()

    videos = list_videos(args.videos)
    if not videos:
        print(f"no videos found in {args.videos}", file=sys.stderr)
        return 1
    preds, report = {}, {}
    outdir = Path(args.outdir)
    for v in videos:
        t0 = time.perf_counter()
        analysis = None
        if args.visualize:
            analysis = analyze_video(str(v))
            segs = analysis.segments()
        else:
            segs = solution.detect_events(str(v))
        t_a = time.perf_counter() - t0
        preds[v.name] = {"events": segs, "risk": []}
        meta = probe(str(v))
        entry = {"duration_sec": round(meta.duration, 3), "part_a_sec": round(t_a, 2), "n_events": len(segs),
                 "problems": validate_segments(segs, meta.duration if meta.duration > 0 else None)}
        risk = None
        if args.risk or args.visualize:
            t1 = time.perf_counter()
            est = solution.RiskEstimator()
            est.reset(meta.as_dict())
            cap = cv2.VideoCapture(str(v))
            ts, rs, i = [], [], 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = i / meta.fps
                rs.append(est.step(frame, t))
                ts.append(t)
                i += 1
            cap.release()
            risk = (np.array(ts), np.array(rs))
            preds[v.name]["risk"] = [[round(a, 4), round(b, 4)] for a, b in zip(ts, rs)]
            entry["part_b_sec"] = round(time.perf_counter() - t1, 2)
            entry["max_risk"] = round(float(risk[1].max()) if len(rs) else 0.0, 3)
            (outdir / "risk").mkdir(parents=True, exist_ok=True)
            with open(outdir / "risk" / f"{v.stem}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_sec", "risk"])
                w.writerows([[f"{a:.3f}", f"{b:.4f}"] for a, b in zip(ts, rs)])
        if args.visualize and analysis is not None:
            vis = outdir / "vis"
            plot_timeline(analysis.events, analysis.duration, str(vis / f"{v.stem}_timeline.png"), f"{v.name}", risk)
            render_annotated_video(str(v), analysis, str(vis / f"{v.stem}_annotated.mp4"), risk)
            with open(vis / f"{v.stem}_events.json", "w") as fh:
                json.dump({"video": v.name, "duration": analysis.duration, "events": [e.as_dict() for e in analysis.events],
                           "candidates": [e.as_dict() for e in analysis.raw_events], "warnings": analysis.warnings,
                           "timings": analysis.timings}, fh, indent=2)
        total = entry["part_a_sec"] + entry.get("part_b_sec", 0.0)
        entry["runtime_ratio"] = round(total / meta.duration, 3) if meta.duration > 0 else None
        report[v.name] = entry
        print(f"{v.name}: {len(segs)} events, A {t_a:.1f}s" + (f", B {entry['part_b_sec']:.1f}s" if "part_b_sec" in entry else "")
              + f", video {meta.duration:.1f}s", file=sys.stderr)
    with open(args.out, "w") as fh:
        json.dump({"team": args.team, "videos": preds}, fh, indent=1)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "run_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    bad = {k: v["problems"] for k, v in report.items() if v["problems"]}
    print(f"wrote {args.out} ({len(preds)} videos); report -> {outdir / 'run_report.json'}", file=sys.stderr)
    if bad:
        print(f"INVALID OUTPUT: {bad}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
