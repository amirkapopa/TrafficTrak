"""Produce candidate event clips for manual review, and turn reviews into dev labels.

Step 1 - mine candidates with deliberately loose thresholds:
    python scripts/annotate_candidates.py mine --videos samples/
  -> outputs/candidates/<video>/<id>_<label>.mp4   (annotated clip, +-pad seconds)
     outputs/candidates/manifest.json
     outputs/candidates/review.csv                 (one row per candidate)

Step 2 - a human fills review.csv: accept = y/n, optionally corrects label /
start / end (seconds in the FULL video), and may add rows with
candidate_id = manual for events the system missed.

Step 3 - build development labels in the official ground-truth format:
    python scripts/annotate_candidates.py labels --review outputs/candidates/review.csv --videos samples/
  -> labels/dev_labels.json   ({video: {"duration": s, "fps": f, "events": [[start, end, label], ...]}})
Every video in --videos is treated as reviewed (no accepted rows -> "events": []).
Score with:  python evaluate.py --pred predictions_samples.json --gt labels/dev_labels.json --per-video

Only human-accepted rows become labels: unlabelled sample videos are never
treated as ground truth.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
from _bootstrap import ROOT, list_videos

from src.config import load_config
from src.pipeline import analyze_video
from src.segments import CLASSES, RawEvent, finalize, to_output
from src.video import probe
from src.visualize import C_ALERT, C_TRACK, _label, to_browser_mp4

LOOSE = {
    "classes": {
        "stopped_vehicle": {"min_duration_sec": 6.0, "passers_min": 1, "isolated_min_duration_sec": 20.0},
        "wrong_way": {"min_duration_sec": 1.0, "auto_min_support": 4, "auto_min_concentration": 0.75},
        "near_miss": {"ttc_threshold": 1.5, "decel_threshold": 1.5, "min_closing_speed": 1.0},
        "accident": {"decel_drop": 0.7, "struck_velocity_jolt": 0.2},
        "congestion": {"min_duration_sec": 15.0, "min_vehicles": 4},
        "jaywalking": {"min_duration_sec": 0.5},
    },
    "segments": {"min_duration_sec": {"default": 0.3, "stopped_vehicle": 6.0, "congestion": 10.0}},
}
FIELDS = ["video", "candidate_id", "label", "start", "end", "tracks", "note", "clip",
          "accept", "label_fixed", "start_fixed", "end_fixed", "comment"]


def write_clip(video: Path, analysis, ev: RawEvent, path: Path, pad: float) -> None:
    meta = analysis.meta
    cap = cv2.VideoCapture(str(video))
    f0 = max(0, int((ev.start - pad) * meta.fps))
    f1 = int((ev.end + pad) * meta.fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    scale = min(1.0, 960 / meta.width)
    size = (int(meta.width * scale), int(meta.height * scale))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, size)
    involved = {s.tid: s for s in analysis.series if s.tid in ev.tracks}
    for f in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        t = f / meta.fps
        for s in involved.values():
            i = s.local(int(round(t / analysis.dt)))
            if i >= 0:
                x0, y0, x1, y1 = s.box[i]
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), C_ALERT if ev.start <= t <= ev.end else C_TRACK, 2)
                _label(frame, f"{s.tid} {s.cls}", (x0, y0))
        active = ev.start <= t <= ev.end
        _label(frame, f"{ev.label} {'ACTIVE' if active else ''} t={t:.2f}s [{ev.start:.2f}-{ev.end:.2f}]", (10, 30),
               (255, 255, 255), 0.7, C_ALERT if active else (60, 60, 60))
        vw.write(cv2.resize(frame, size) if scale < 1 else frame)
    cap.release()
    vw.release()
    to_browser_mp4(str(path))


def mine(args) -> int:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(overrides=LOOSE)
    manifest, rows = {}, []
    for v in list_videos(args.videos):
        analysis = analyze_video(str(v), cfg=cfg)
        vdir = out / v.stem
        vdir.mkdir(exist_ok=True)
        cands = sorted(analysis.raw_events, key=lambda e: (e.start, e.label))
        manifest[v.name] = []
        for i, ev in enumerate(cands):
            clip = vdir / f"{i:03d}_{ev.label}.mp4"
            if not args.no_clips:
                write_clip(v, analysis, ev, clip, args.pad)
            manifest[v.name].append({"id": i, **ev.as_dict(), "clip": str(clip.relative_to(out))})
            rows.append({"video": v.name, "candidate_id": i, "label": ev.label, "start": f"{ev.start:.2f}",
                         "end": f"{ev.end:.2f}", "tracks": " ".join(map(str, ev.tracks)), "note": ev.note,
                         "clip": str(clip.relative_to(out))})
        print(f"{v.name}: {len(cands)} candidates", file=sys.stderr)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(out / "review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out / 'review.csv'} ({len(rows)} rows); fill in `accept` and corrections", file=sys.stderr)
    return 0


def labels(args) -> int:
    per_video: dict[str, list[RawEvent]] = {}
    with open(args.review, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("accept") or "").strip().lower() not in ("y", "yes", "1", "true"):
                continue
            label = (r.get("label_fixed") or r.get("label") or "").strip()
            if label not in CLASSES:
                print(f"skipping row with unknown label {label!r}", file=sys.stderr)
                continue
            try:
                s = float(r.get("start_fixed") or r["start"])
                e = float(r.get("end_fixed") or r["end"])
            except (TypeError, ValueError):
                print(f"skipping row without valid start/end: {r}", file=sys.stderr)
                continue
            per_video.setdefault(r["video"], []).append(RawEvent(label, s, e))
    cfg = {"segments": {"merge_gap_sec": {"default": 0.0}, "min_duration_sec": {"default": 0.0}}}
    out = {}
    for v in list_videos(args.videos):
        meta = probe(str(v))
        if not meta.ok:
            print(f"skipping unreadable {v.name}", file=sys.stderr)
            continue
        out[v.name] = {"duration": round(meta.duration, 3), "fps": round(meta.fps, 3),
                       "events": to_output(finalize(per_video.pop(v.name, []), meta.duration, cfg))}
    for name in per_video:
        print(f"warning: {name} is in the review file but not in {args.videos} - skipped", file=sys.stderr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    n = sum(len(v["events"]) for v in out.values())
    print(f"-> {args.out}: {n} human-verified events in {len(out)} videos", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--videos", nargs="+", default=[str(ROOT / "samples")])
    m.add_argument("--outdir", default=str(ROOT / "outputs" / "candidates"))
    m.add_argument("--pad", type=float, default=2.0)
    m.add_argument("--no-clips", action="store_true")
    lb = sub.add_parser("labels")
    lb.add_argument("--review", default=str(ROOT / "outputs" / "candidates" / "review.csv"))
    lb.add_argument("--out", default=str(ROOT / "labels" / "dev_labels.json"))
    lb.add_argument("--videos", nargs="+", default=[str(ROOT / "samples")])
    args = ap.parse_args()
    return mine(args) if args.cmd == "mine" else labels(args)


if __name__ == "__main__":
    sys.exit(main())
