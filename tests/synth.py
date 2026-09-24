"""Synthetic trajectories for rule tests (no video, no detector needed)."""

from __future__ import annotations

import numpy as np

from src.config import load_config
from src.geometry import build_geometry
from src.pipeline import analyze_tracks
from src.rules import run_rules
from src.segments import finalize
from src.tracking import KalmanBox, Track

W, H = 1280, 720
RATE = 15.0
DT = 1.0 / RATE


def make_track(tid, cls, path_fn, t0, t1, size=(80, 50), score=0.9, size_fn=None):
    """path_fn(t) -> (cx, ground_y) in pixels."""
    tr = Track(tid, KalmanBox(np.array([0, 0, 10, 10], dtype=np.float32)), start_t=t0, last_t=t1, last_frame=0,
               activated=True)
    for t in np.arange(t0, t1 + 1e-9, DT):
        cx, gy = path_fn(float(t))
        w, h = size_fn(float(t)) if size_fn else size
        tr.record(float(t), int(round(t * RATE)), np.array([cx - w / 2, gy - h, cx + w / 2, gy], np.float32), score, cls)
    return tr


def linear(p0, v, t_start):
    return lambda t: (p0[0] + v[0] * (t - t_start), p0[1] + v[1] * (t - t_start))


def piecewise(points):
    """points: list of (t, x, y); linear interpolation, clamped at ends."""
    ts = np.array([p[0] for p in points], float)
    xs = np.array([p[1] for p in points], float)
    ys = np.array([p[2] for p in points], float)
    return lambda t: (float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys)))


def run(tracks, duration, geometry=None, overrides=None, signals=None, prior=None):
    cfg = load_config(overrides=overrides)
    geo = build_geometry(geometry or {}, W, H)
    ctx = analyze_tracks(tracks, cfg, geo, W, H, DT, duration, signals=signals, prior_flow=prior)
    raw = run_rules(ctx)
    return finalize(raw, duration, cfg), raw, ctx


def labels(events):
    return [e.label for e in events]


def of(events, label):
    return [e for e in events if e.label == label]


def norm(pts):
    return [[x / W, y / H] for x, y in pts]
