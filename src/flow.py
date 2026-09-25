"""Learned scene model for a fixed camera: traffic-direction field and
carriageway occupancy, estimated from moving-vehicle trajectories.

Why: the camera never moves, so the dominant direction of traffic in each
image cell is a stable scene fact.  It can be (a) learned per video from the
video's own tracks (leave-one-out, so a wrong-way driver never votes for its
own direction) and (b) accumulated over all sample videos into a prior
(``scripts/build_scene_prior.py`` -> ``config/scene_prior.npz``).  Both are
used only when no hand-calibrated lanes/carriageway exist.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from .features import TrackSeries

log = logging.getLogger("traffictrak.flow")

FLOW_GROUPS = ("vehicle", "two_wheeler")


class FlowField:
    def __init__(self, grid=(32, 18), occ_grid=(64, 36), bins: int = 16):
        self.gw, self.gh = int(grid[0]), int(grid[1])
        self.ow, self.oh = int(occ_grid[0]), int(occ_grid[1])
        self.bins = int(bins)
        self.hist = np.zeros((self.gh, self.gw, self.bins), dtype=np.float64)
        self.support = np.zeros((self.gh, self.gw), dtype=np.float64)
        self.occ = np.zeros((self.oh, self.ow), dtype=np.float64)
        # distinct vehicles that stood still (>= min_stop_sec) in each fine cell:
        # signal queues, bus stops, parking - places where stopping is normal
        self.stop_occ = np.zeros((self.oh, self.ow), dtype=np.float64)
        self._stop_contrib: dict[int, np.ndarray] = {}
        self._contrib: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._occ_dil: np.ndarray | None = None
        self.rel_frac = 0.25  # carriageway = at least this share of the busiest (p90) cells

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: dict) -> FlowField:
        sc = cfg.get("scene", {})
        return cls(tuple(sc.get("flow_grid", (32, 18))), tuple(sc.get("occupancy_grid", (64, 36))),
                   int(sc.get("flow_bins", 16)))

    def _cells(self, xn: np.ndarray, yn: np.ndarray, gw: int, gh: int) -> tuple[np.ndarray, np.ndarray]:
        cx = np.clip((xn * gw).astype(np.int64), 0, gw - 1)
        cy = np.clip((yn * gh).astype(np.int64), 0, gh - 1)
        return cy, cx

    def _angle_bin(self, ang: np.ndarray) -> np.ndarray:
        return np.floor(((ang % (2 * math.pi)) / (2 * math.pi)) * self.bins).astype(np.int64) % self.bins

    def add_series(self, s: TrackSeries, width: int, height: int, moving_speed: float = 0.4) -> None:
        """Add one track's votes (each track contributes <= 1 vote per cell)."""
        if s.group not in FLOW_GROUPS or s.n < 2:
            return
        mov = s.speed_n >= moving_speed
        if mov.sum() < 2:
            return
        xn, yn = s.gx[mov] / width, s.gy[mov] / height
        cy, cx = self._cells(xn, yn, self.gw, self.gh)
        b = self._angle_bin(s.heading[mov])
        flat = cy * self.gw + cx
        cells = np.unique(flat)
        votes = np.zeros((len(cells), self.bins))
        pos = np.searchsorted(cells, flat)
        np.add.at(votes, (pos, b), 1.0)
        votes /= votes.sum(axis=1, keepdims=True)
        ccy, ccx = cells // self.gw, cells % self.gw
        self.hist[ccy, ccx] += votes
        self.support[ccy, ccx] += 1.0
        oy, ox = self._cells(xn, yn, self.ow, self.oh)
        ocells = np.unique(oy * self.ow + ox)
        self.occ[ocells // self.ow, ocells % self.ow] += 1.0
        self._contrib[s.tid] = (cells, votes, ocells)
        self._occ_dil = None

    def add_stops(self, s: TrackSeries, width: int, height: int, stationary_speed: float, min_stop_sec: float) -> None:
        """Record where this vehicle stood still for at least ``min_stop_sec``."""
        if s.group not in FLOW_GROUPS or s.n < 2:
            return
        still = s.speed_n < stationary_speed
        if still.sum() * s.dt < min_stop_sec:
            return
        oy, ox = self._cells(s.gx[still] / width, s.gy[still] / height, self.ow, self.oh)
        flat, counts = np.unique(oy * self.ow + ox, return_counts=True)
        cells = flat[counts * s.dt >= min_stop_sec]
        if cells.size == 0:
            return
        self.stop_occ[cells // self.ow, cells % self.ow] += 1.0
        self._stop_contrib[s.tid] = cells

    def stop_count(self, x: float, y: float, width: int, height: int, exclude: int | None = None) -> float:
        """How many *other* vehicles stood still within one fine cell of (x, y)."""
        oy, ox = self._cells(np.array([x / width]), np.array([y / height]), self.ow, self.oh)
        oy, ox = int(oy[0]), int(ox[0])
        y0, y1, x0, x1 = max(oy - 1, 0), min(oy + 2, self.oh), max(ox - 1, 0), min(ox + 2, self.ow)
        count = float(self.stop_occ[y0:y1, x0:x1].max())
        if exclude is not None and exclude in self._stop_contrib:
            own = self._stop_contrib[exclude]
            ys, xs = own // self.ow, own % self.ow
            if np.any((ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)):
                count -= 1.0
        return max(count, 0.0)

    def add_prior(self, other: FlowField, weight: float = 1.0) -> None:
        if (other.gw, other.gh, other.bins) != (self.gw, self.gh, self.bins) or (other.ow, other.oh) != (self.ow, self.oh):
            log.warning("scene prior grid mismatch - ignored")
            return
        self.hist += weight * other.hist
        self.support += weight * other.support
        self.occ += weight * other.occ
        self.stop_occ += weight * other.stop_occ
        self._occ_dil = None

    # ------------------------------------------------------------------ query
    def _cell_hist(self, cy: int, cx: int, exclude: int | None) -> tuple[np.ndarray, float]:
        h = self.hist[cy, cx].copy()
        sup = float(self.support[cy, cx])
        if exclude is not None and exclude in self._contrib:
            cells, votes, _ = self._contrib[exclude]
            flat = cy * self.gw + cx
            j = np.searchsorted(cells, flat)
            if j < len(cells) and cells[j] == flat:
                h -= votes[j]
                sup -= 1.0
        return np.clip(h, 0, None), max(sup, 0.0)

    def dominant(self, x: float, y: float, width: int, height: int, exclude: int | None = None):
        """Return (angle, concentration, support) of the traffic direction at
        pixel (x, y).  concentration = share of votes within +-45 degrees."""
        cy, cx = self._cells(np.array([x / width]), np.array([y / height]), self.gw, self.gh)
        h, sup = self._cell_hist(int(cy[0]), int(cx[0]), exclude)
        total = h.sum()
        if total <= 0:
            return 0.0, 0.0, 0.0
        sm = 0.5 * h + 0.25 * np.roll(h, 1) + 0.25 * np.roll(h, -1)
        peak = int(np.argmax(sm))
        half = max(1, self.bins // 8)
        idx = [(peak + d) % self.bins for d in range(-half, half + 1)]
        mass = h[idx]
        centres = (np.array(idx) + 0.5) * 2 * math.pi / self.bins
        ang = math.atan2(float(np.sum(mass * np.sin(centres))), float(np.sum(mass * np.cos(centres))))
        return ang, float(mass.sum() / total), sup

    def _occ_dilated(self) -> np.ndarray:
        if self._occ_dil is None:
            self._occ_dil = maximum_filter(self.occ, size=3, mode="nearest")
        return self._occ_dil

    def carriageway(self, x: float, y: float, width: int, height: int, min_tracks: float,
                    exclude: int | None = None) -> bool:
        """True if at least ``min_tracks`` other moving vehicles passed here."""
        oy, ox = self._cells(np.array([x / width]), np.array([y / height]), self.ow, self.oh)
        oy, ox = int(oy[0]), int(ox[0])
        count = float(self._occ_dilated()[oy, ox])
        if exclude is not None and exclude in self._contrib:
            own = self._contrib[exclude][2]
            ys, xs = own // self.ow, own % self.ow
            if np.any((np.abs(ys - oy) <= 1) & (np.abs(xs - ox) <= 1)):
                count -= 1.0
        return count >= self.road_threshold(min_tracks)

    def road_threshold(self, min_tracks: float) -> float:
        """Absolute floor, raised to ``rel_frac`` x p90 of occupied cells so the
        threshold scales when a multi-video prior inflates all counts (kerb-side
        cells next to a busy lane stay off-road)."""
        occ = self._occ_dilated()
        nz = occ[occ > 0]
        rel = self.rel_frac * float(np.percentile(nz, 90)) if nz.size else 0.0
        return max(float(min_tracks), rel)

    def carriageway_mask(self, min_tracks: float) -> np.ndarray:
        return self._occ_dilated() >= self.road_threshold(min_tracks)

    def direction_groups(self, min_share: float = 0.12, min_sep_deg: float = 60.0) -> list[float]:
        """Principal traffic directions of the scene (angles, radians)."""
        glob = self.hist.sum(axis=(0, 1))
        total = glob.sum()
        if total <= 0:
            return []
        sm = 0.5 * glob + 0.25 * np.roll(glob, 1) + 0.25 * np.roll(glob, -1)
        sep_bins = max(1, int(round(min_sep_deg / 360.0 * self.bins)))
        peaks: list[int] = []
        for b in np.argsort(-sm, kind="stable"):
            b = int(b)
            if sm[b] / total < min_share:
                break
            if all(min(abs(b - p), self.bins - abs(b - p)) >= sep_bins for p in peaks):
                peaks.append(b)
        return [((p + 0.5) * 2 * math.pi / self.bins + math.pi) % (2 * math.pi) - math.pi for p in sorted(peaks)]

    @property
    def n_tracks(self) -> int:
        return len(self._contrib)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, hist=self.hist, support=self.support, occ=self.occ, stop_occ=self.stop_occ,
                            grid=np.array([self.gw, self.gh]), occ_grid=np.array([self.ow, self.oh]),
                            bins=np.array([self.bins]))

    @classmethod
    def load(cls, path: str | Path) -> FlowField | None:
        try:
            d = np.load(path)
            f = cls(tuple(d["grid"].tolist()), tuple(d["occ_grid"].tolist()), int(d["bins"][0]))
            f.hist, f.support, f.occ = d["hist"].astype(np.float64), d["support"].astype(np.float64), d["occ"].astype(np.float64)
            if "stop_occ" in d.files:
                f.stop_occ = d["stop_occ"].astype(np.float64)
            return f
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot load scene prior %s: %s", path, exc)
            return None


def nearest_group(angle: float, groups: list[float], max_diff_deg: float = 60.0) -> int:
    """Index of the direction group closest to ``angle`` or -1."""
    best, best_d = -1, math.radians(max_diff_deg)
    for i, g in enumerate(groups):
        d = abs((angle - g + math.pi) % (2 * math.pi) - math.pi)
        if d <= best_d:
            best, best_d = i, d
    return best
