"""Exploratory analysis of the sample videos + scene-prior builder.

    python scripts/analyze_samples.py --videos samples/                 # EDA only
    python scripts/analyze_samples.py --videos samples/ --write-prior   # + config/scene_prior.npz, background

Outputs (outputs/eda/):
  reference_median.jpg        empty-road reference frame (median of per-video backgrounds)
  heatmap_vehicles.jpg / heatmap_people.jpg   ground-contact heat maps
  flow_field.jpg              learned dominant traffic direction per cell + learned carriageway
  traffic_light_candidates.jpg + suggested normalised signal ROIs (in eda.md)
  geometry_overlay.jpg        current config/camera_geometry.yaml over the reference frame
  <video>/first.jpg, background.jpg, flow.jpg, heatmap.jpg
  eda.json, eda.md            metadata, class counts, speeds, track statistics

The prior only contains stable scene facts (where and in which direction
traffic moves, what the empty road looks like); it never stores events.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from _bootstrap import ROOT, list_videos

from src.config import geometry_config_path, load_config, resolve_path
from src.detection import COCO_NAMES, get_detector
from src.features import build_all_series
from src.flow import FlowField
from src.geometry import load_geometry
from src.pipeline import choose_stride
from src.tracking import ByteTracker, iou_matrix, stitch_stationary
from src.video import FrameReader, probe
from src.visualize import draw_flow, draw_geometry, heatmap_overlay

TL = COCO_NAMES.index("traffic_light")


def cluster_boxes(boxes: list, min_count: int) -> list[tuple[np.ndarray, int]]:
    clusters: list[list[np.ndarray]] = []
    for b in boxes:
        for c in clusters:
            if iou_matrix(np.array([b]), np.array([np.median(c, axis=0)]))[0, 0] > 0.3:
                c.append(b)
                break
        else:
            clusters.append([b])
    out = [(np.median(c, axis=0), len(c)) for c in clusters if len(c) >= min_count]
    return sorted(out, key=lambda x: -x[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", default=[str(ROOT / "samples")])
    ap.add_argument("--outdir", default=str(ROOT / "outputs" / "eda"))
    ap.add_argument("--max-seconds", type=float, default=None, help="analyse at most this many seconds per video")
    ap.add_argument("--write-prior", action="store_true")
    args = ap.parse_args()
    videos = list_videos(args.videos)
    if not videos:
        print(f"no videos in {args.videos}", file=sys.stderr)
        return 1
    cfg = load_config()
    det = get_detector(cfg)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    global_flow = FlowField.from_config(cfg)
    moving = float(cfg["features"]["moving_speed"])
    backgrounds, veh_pts, ped_pts, tl_boxes = [], [], [], []
    summary = {"videos": {}, "stride": None}
    ref_size = None
    n_sampled_frames = 0
    for v in videos:
        meta = probe(str(v))
        if not meta.ok:
            summary["videos"][v.name] = {"error": meta.error}
            continue
        ref_size = ref_size or (meta.width, meta.height)
        vdir = out / v.stem
        vdir.mkdir(exist_ok=True)
        stride = choose_stride(cfg, det.on_gpu)
        summary["stride"] = stride
        tracker = ByteTracker(cfg, meta.fps)
        bg_frames, first = [], None
        bg_every = max(1.0, (meta.duration or 60) / 40)
        next_bg = 0.0
        det_counts: Counter = Counter()
        for idx, t, frame in FrameReader(str(v), meta.fps, stride, meta.n_frames):
            if args.max_seconds and t > args.max_seconds:
                break
            if first is None:
                first = frame.copy()
            d = det(frame)
            tracker.update(d, t, idx)
            det_counts.update(d.names())
            if t >= next_bg:
                next_bg = t + bg_every
                bg_frames.append(cv2.resize(frame, ref_size))
                n_sampled_frames += 1
                tl_boxes += [b * np.array([ref_size[0] / meta.width, ref_size[1] / meta.height] * 2)
                             for b, c, s in zip(d.boxes, d.classes, d.scores) if c == TL and s >= 0.3]
        if first is None:
            summary["videos"][v.name] = {"error": "no frames"}
            continue
        dt = stride / meta.fps
        tracks = stitch_stationary(tracker.all_tracks(), 8.0, 0.45)
        series = build_all_series(tracks, dt, cfg)
        flow = FlowField.from_config(cfg)
        for s in series:
            flow.add_series(s, meta.width, meta.height, moving)
            global_flow.add_series(s, meta.width, meta.height, moving)
            pts = s.ground() * np.array([ref_size[0] / meta.width, ref_size[1] / meta.height])
            if s.group in ("vehicle", "two_wheeler"):
                veh_pts.extend(pts[::3])
            elif s.group == "person":
                ped_pts.extend(pts[::3])
        bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8)
        backgrounds.append(bg)
        cv2.imwrite(str(vdir / "first.jpg"), first)
        cv2.imwrite(str(vdir / "background.jpg"), bg)
        cv2.imwrite(str(vdir / "flow.jpg"), draw_flow(cv2.resize(bg, (meta.width, meta.height)), flow))
        vp = np.concatenate([s.ground() for s in series if s.group in ("vehicle", "two_wheeler")] or [np.zeros((0, 2))])
        cv2.imwrite(str(vdir / "heatmap.jpg"), heatmap_overlay(cv2.resize(bg, (meta.width, meta.height)), vp))
        cls_tracks = Counter(s.cls for s in series)
        speeds = {c: round(float(np.median(np.concatenate([s.speed_n for s in series if s.cls == c]))), 3)
                  for c in cls_tracks}
        summary["videos"][v.name] = {
            "fps": meta.fps, "fps_reliable": meta.fps_reliable, "width": meta.width, "height": meta.height,
            "n_frames": meta.n_frames, "duration_sec": round(meta.duration, 2),
            "tracks_per_class": dict(sorted(cls_tracks.items())), "median_speed_scale_per_s": speeds,
            "detections_per_class": dict(sorted(det_counts.items())),
            "direction_groups_deg": [round(float(np.degrees(a)), 1) for a in flow.direction_groups()],
        }
        print(f"{v.name}: {len(series)} tracks {dict(cls_tracks)}", file=sys.stderr)
    if not backgrounds:
        print("no readable videos", file=sys.stderr)
        return 1
    ref = np.median(np.stack(backgrounds), axis=0).astype(np.uint8)
    cv2.imwrite(str(out / "reference_median.jpg"), ref)
    cv2.imwrite(str(out / "heatmap_vehicles.jpg"), heatmap_overlay(ref, np.array(veh_pts).reshape(-1, 2)))
    cv2.imwrite(str(out / "heatmap_people.jpg"), heatmap_overlay(ref, np.array(ped_pts).reshape(-1, 2)))
    cv2.imwrite(str(out / "flow_field.jpg"), draw_flow(ref, global_flow))
    geo = load_geometry(geometry_config_path(), ref.shape[1], ref.shape[0])
    cv2.imwrite(str(out / "geometry_overlay.jpg"), draw_geometry(ref, geo))
    tl_img = ref.copy()
    candidates = []
    for box, count in cluster_boxes(tl_boxes, max(3, int(0.2 * n_sampled_frames))):
        x0, y0, x1, y1 = box
        pw, ph = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
        roi = [(x0 - pw) / ref.shape[1], (y0 - ph) / ref.shape[0], (x1 + pw) / ref.shape[1], (y1 + ph) / ref.shape[0]]
        candidates.append({"roi_normalised": [round(float(r), 4) for r in roi], "support": int(count)})
        cv2.rectangle(tl_img, (int(x0), int(y0)), (int(x1), int(y1)), (200, 60, 200), 2)
    cv2.imwrite(str(out / "traffic_light_candidates.jpg"), tl_img)
    summary["traffic_light_candidates"] = candidates
    summary["global_direction_groups_deg"] = [round(float(np.degrees(a)), 1) for a in global_flow.direction_groups()]
    summary["flow_tracks"] = global_flow.n_tracks
    with open(out / "eda.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    lines = ["# Sample-video EDA", "", "| video | duration s | fps | size | tracks per class |", "|---|---|---|---|---|"]
    for name, s in summary["videos"].items():
        if "error" in s:
            lines.append(f"| {name} | error: {s['error']} | | | |")
        else:
            lines.append(f"| {name} | {s['duration_sec']} | {s['fps']:.2f} | {s['width']}x{s['height']} | {s['tracks_per_class']} |")
    lines += ["", f"Principal traffic directions (deg, image frame, 0 = right, 90 = down): {summary['global_direction_groups_deg']}",
              "", "## Traffic-light candidates (paste into `signals:` after checking the image)", ""]
    lines += [f"- roi: {c['roi_normalised']}  (seen {c['support']}x)" for c in candidates] or ["- none detected"]
    (out / "eda.md").write_text("\n".join(lines) + "\n")
    if args.write_prior:
        sc = cfg["scene"]
        prior = resolve_path(sc["prior_path"])
        global_flow.save(prior)
        cv2.imwrite(str(resolve_path(sc["background_path"])), ref)
        print(f"wrote scene prior {prior} ({global_flow.n_tracks} tracks) and background reference", file=sys.stderr)
    print(f"EDA written to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
