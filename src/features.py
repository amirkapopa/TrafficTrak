"""Track kinematics on a uniform time grid and pairwise interaction features.

All tracks are resampled to a global grid ``t_k = k * dt`` (``dt`` = base
frame stride / fps) so pairwise features reduce to array intersections.
Speeds are normalised by the object's own size (``scale = sqrt(w*h)``), which
makes thresholds approximately perspective-invariant for a fixed camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .detection import CLASS_GROUP
from .tracking import Track


# --------------------------------------------------------------------------
# boolean-run helpers (used everywhere in the rules)
# --------------------------------------------------------------------------
def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Close False gaps of length <= max_gap that lie between True values."""
    m = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or m.size == 0:
        return m
    rs = runs(m)
    for (_, a1), (b0, _) in zip(rs[:-1], rs[1:]):
        if b0 - a1 - 1 <= max_gap:
            m[a1 + 1:b0] = True
    return m


def drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).copy()
    for a, b in runs(m):
        if b - a + 1 < min_len:
            m[a:b + 1] = False
    return m


def sustained_onset(mask: np.ndarray, start: int, min_len: int) -> int:
    """First index >= start where mask stays True for min_len samples, or -1."""
    m = np.asarray(mask, dtype=bool)
    count = 0
    for i in range(max(start, 0), len(m)):
        count = count + 1 if m[i] else 0
        if count >= min_len:
            return i - min_len + 1
    return -1


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(x) < 3:
        return np.asarray(x, dtype=np.float64)
    return gaussian_filter1d(np.asarray(x, dtype=np.float64), sigma, axis=0, mode="nearest")


# --------------------------------------------------------------------------
# per-track series
# --------------------------------------------------------------------------
@dataclass
class TrackSeries:
    tid: int
    cls: str
    group: str
    dt: float
    k0: int
    box: np.ndarray          # (n, 4) smoothed xyxy
    box_raw: np.ndarray      # (n, 4) unsmoothed (interpolated) xyxy - used for contact tests
    observed: np.ndarray     # (n,) True where a real detection exists
    score: np.ndarray        # (n,)
    gx: np.ndarray           # ground point (bottom-centre), smoothed
    gy: np.ndarray
    scale: np.ndarray        # sqrt(w*h), smoothed (px)
    vx: np.ndarray           # px/s
    vy: np.ndarray
    speed_n: np.ndarray      # scale units / s
    heading: np.ndarray      # rad, raw (noisy when slow)
    heading_filled: np.ndarray  # rad, held through slow samples
    heading_rate: np.ndarray    # rad/s, 0 when slow
    accel_n: np.ndarray      # d(speed_n)/dt

    @property
    def n(self) -> int:
        return len(self.gx)

    @property
    def k1(self) -> int:
        return self.k0 + self.n - 1

    @property
    def t(self) -> np.ndarray:
        return (self.k0 + np.arange(self.n)) * self.dt

    @property
    def t0(self) -> float:
        return self.k0 * self.dt

    @property
    def t1(self) -> float:
        return self.k1 * self.dt

    def time(self, i: int) -> float:
        return (self.k0 + i) * self.dt

    def local(self, k: int) -> int:
        i = k - self.k0
        return i if 0 <= i < self.n else -1

    def ground(self) -> np.ndarray:
        return np.stack([self.gx, self.gy], 1)

    @property
    def w(self) -> np.ndarray:
        return self.box[:, 2] - self.box[:, 0]

    @property
    def h(self) -> np.ndarray:
        return self.box[:, 3] - self.box[:, 1]

    def footprint(self, band: float, raw: bool = False) -> np.ndarray:
        b = self.box_raw if raw else self.box
        return np.stack([b[:, 0], b[:, 3] - band * (b[:, 3] - b[:, 1]), b[:, 2], b[:, 3]], 1)

    def path_length(self, i0: int, i1: int) -> float:
        """Path length between two indices in scale units."""
        if i1 <= i0:
            return 0.0
        g = self.ground()[i0:i1 + 1]
        seg = np.linalg.norm(np.diff(g, axis=0), axis=1)
        sc = 0.5 * (self.scale[i0:i1] + self.scale[i0 + 1:i1 + 1])
        return float(np.sum(seg / np.maximum(sc, 1.0)))


def build_series(track: Track, dt: float, cfg: dict) -> TrackSeries | None:
    """Resample a track's observations onto the global grid and derive kinematics."""
    if not track.t_hist:
        return None
    fc = cfg.get("features", {})
    moving = float(fc.get("moving_speed", 0.4))
    t_obs = np.asarray(track.t_hist, dtype=np.float64)
    k_obs = np.rint(t_obs / dt).astype(np.int64)
    boxes = np.stack(track.box_hist).astype(np.float64)
    scores = np.asarray(track.score_hist, dtype=np.float64)
    # keep the last observation per grid cell (sorted, deterministic)
    order = np.argsort(k_obs, kind="stable")
    k_obs, boxes, scores = k_obs[order], boxes[order], scores[order]
    uniq_last = np.r_[k_obs[1:] != k_obs[:-1], True]
    k_obs, boxes, scores = k_obs[uniq_last], boxes[uniq_last], scores[uniq_last]
    k0, k1 = int(k_obs[0]), int(k_obs[-1])
    ks = np.arange(k0, k1 + 1)
    n = len(ks)
    box = np.stack([np.interp(ks, k_obs, boxes[:, j]) for j in range(4)], 1)
    score = np.interp(ks, k_obs, scores)
    observed = np.zeros(n, dtype=bool)
    observed[k_obs - k0] = True

    sigma = float(fc.get("smooth_sigma_sec", 0.35)) / dt
    box_s = smooth(box, sigma)
    gx = (box_s[:, 0] + box_s[:, 2]) / 2
    gy = box_s[:, 3]
    w = np.maximum(box_s[:, 2] - box_s[:, 0], 1.0)
    h = np.maximum(box_s[:, 3] - box_s[:, 1], 1.0)
    scale = np.maximum(np.sqrt(w * h), 4.0)
    if n >= 2:
        vx = np.gradient(gx, dt)
        vy = np.gradient(gy, dt)
    else:
        vx = np.zeros(n)
        vy = np.zeros(n)
    speed_n = np.hypot(vx, vy) / scale
    heading = np.arctan2(vy, vx)
    mov = speed_n >= moving
    filled = heading.copy()
    if mov.any():
        idx = np.where(mov, np.arange(n), -1)
        idx = np.maximum.accumulate(idx)
        first = int(np.argmax(mov))
        idx[idx < 0] = first
        filled = heading[idx]
    unwrapped = np.unwrap(filled)
    rate = np.gradient(smooth(unwrapped, sigma * 0.5), dt) if n >= 2 else np.zeros(n)
    rate = np.where(mov, rate, 0.0)
    accel = np.gradient(smooth(speed_n, sigma * 0.5), dt) if n >= 2 else np.zeros(n)
    cls = track.cls_name
    return TrackSeries(track.track_id, cls, CLASS_GROUP.get(cls, "other"), dt, k0, box_s, box, observed, score,
                       gx, gy, scale, vx, vy, speed_n, heading, filled, rate, accel)


def build_all_series(tracks: list[Track], dt: float, cfg: dict) -> list[TrackSeries]:
    tc = cfg.get("tracker", {})
    min_sec = float(tc.get("min_track_sec", 0.8))
    min_obs = int(tc.get("min_observations", 3))
    out = []
    for tr in sorted(tracks, key=lambda x: x.track_id):
        if len(tr.t_hist) < min_obs or (tr.t_hist[-1] - tr.t_hist[0]) < min_sec:
            continue
        s = build_series(tr, dt, cfg)
        if s is not None:
            out.append(s)
    return out


# --------------------------------------------------------------------------
# pairwise features
# --------------------------------------------------------------------------
@dataclass
class PairFeatures:
    a: TrackSeries
    b: TrackSeries
    k0: int                 # global index of first overlapping sample
    ia: np.ndarray          # local indices into a
    ib: np.ndarray          # local indices into b
    dist: np.ndarray        # ground-point distance / mean scale
    gap: np.ndarray         # smoothed footprint rectangle gap / mean scale (0 = overlap)
    raw_gap: np.ndarray     # same on unsmoothed boxes (brief contacts survive)
    closing: np.ndarray     # -d(dist)/dt  (scale units / s)
    ttc: np.ndarray         # gap / closing (inf when not closing)
    contact: np.ndarray     # raw footprints touch (<= contact_gap) at compatible depth

    @property
    def n(self) -> int:
        return len(self.ia)

    def time(self, j: int) -> float:
        return (self.k0 + j) * self.a.dt


def _rect_gap(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    dx = np.maximum(0.0, np.maximum(ra[:, 0] - rb[:, 2], rb[:, 0] - ra[:, 2]))
    dy = np.maximum(0.0, np.maximum(ra[:, 1] - rb[:, 3], rb[:, 1] - ra[:, 3]))
    return np.hypot(dx, dy)


def pair_features(a: TrackSeries, b: TrackSeries, band: float, depth_tol: float,
                  contact_gap: float = 0.1) -> PairFeatures | None:
    k0, k1 = max(a.k0, b.k0), min(a.k1, b.k1)
    if k1 - k0 < 2:
        return None
    ks = np.arange(k0, k1 + 1)
    ia, ib = ks - a.k0, ks - b.k0
    s = 0.5 * (a.scale[ia] + b.scale[ib])
    dist = np.hypot(a.gx[ia] - b.gx[ib], a.gy[ia] - b.gy[ib]) / s
    fa, fb = a.footprint(band)[ia], b.footprint(band)[ib]
    gap = _rect_gap(fa, fb) / s
    raw_gap = _rect_gap(a.footprint(band, raw=True)[ia], b.footprint(band, raw=True)[ib]) / s
    closing = -np.gradient(smooth(dist, 1.0), a.dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing > 0.05, gap / np.maximum(closing, 1e-6), np.inf)
    min_h = np.minimum(a.h[ia], b.h[ib])
    depth_ok = np.abs(a.box_raw[ia, 3] - b.box_raw[ib, 3]) <= depth_tol * min_h
    contact = (raw_gap <= contact_gap) & depth_ok
    return PairFeatures(a, b, k0, ia, ib, dist, gap, raw_gap, closing, ttc, contact)


def candidate_pairs(series: list[TrackSeries], groups_a: tuple, groups_b: tuple,
                    max_dist: float = 4.0) -> list[tuple[TrackSeries, TrackSeries]]:
    """Pairs that overlap in time and come within ``max_dist`` scale units."""
    out = []
    ordered = sorted(series, key=lambda s: (s.k0, s.tid))
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b.k0 > a.k1:
                break  # sweep: every later track starts after a ends
            if not ((a.group in groups_a and b.group in groups_b) or (b.group in groups_a and a.group in groups_b)):
                continue
            k0, k1 = max(a.k0, b.k0), min(a.k1, b.k1)
            if k1 - k0 < 2:
                continue
            ia, ib = np.arange(k0, k1 + 1) - a.k0, np.arange(k0, k1 + 1) - b.k0
            s = 0.5 * (a.scale[ia] + b.scale[ib])
            d = np.hypot(a.gx[ia] - b.gx[ib], a.gy[ia] - b.gy[ib]) / s
            if float(d.min()) <= max_dist:
                out.append((a, b) if a.tid < b.tid else (b, a))
    return sorted(out, key=lambda p: (p[0].tid, p[1].tid))
