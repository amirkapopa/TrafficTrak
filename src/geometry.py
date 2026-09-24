"""Scene geometry: normalised config -> pixel shapes, containment, crossings,
lane direction tests and projected paths."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("traffictrak.geometry")


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def angle_diff(a, b):
    """Absolute wrapped difference between angles (radians), in [0, pi]."""
    d = (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d)


def to_px(points, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return arr * np.array([width, height], dtype=np.float64)


def to_norm(points, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return arr / np.array([width, height], dtype=np.float64)


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def segments_intersect(p1, p2, q1, q2) -> bool:
    """True if segment p1-p2 intersects segment q1-q2 (touching counts)."""
    p1, p2, q1, q2 = (np.asarray(v, dtype=np.float64) for v in (p1, p2, q1, q2))
    r, s = p2 - p1, q2 - q1
    denom = float(_cross2(r, s))
    qp = q1 - p1
    if abs(denom) < 1e-12:
        if abs(float(_cross2(qp, r))) > 1e-9:
            return False  # parallel
        rr = float(r @ r)
        if rr < 1e-12:
            return bool(np.allclose(p1, q1))
        t0 = float(qp @ r) / rr
        t1 = t0 + float(s @ r) / rr
        return max(t0, t1) >= 0 and min(t0, t1) <= 1
    t = float(_cross2(qp, s)) / denom
    u = float(_cross2(qp, r)) / denom
    return -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9


class Polygon:
    def __init__(self, pts: np.ndarray, name: str = ""):
        self.pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        self.name = name
        if len(self.pts) < 3:
            raise ValueError(f"polygon {name!r} needs >= 3 points")

    def contains(self, pts) -> np.ndarray:
        """Vectorised even-odd point-in-polygon test for an (N, 2) array."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x, y = p[:, 0][:, None], p[:, 1][:, None]
        xi, yi = self.pts[:, 0][None], self.pts[:, 1][None]
        xj, yj = np.roll(self.pts[:, 0], 1)[None], np.roll(self.pts[:, 1], 1)[None]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
        hit = cond & (x < xint)
        return (np.sum(hit, axis=1) % 2) == 1

    def contains_point(self, p) -> bool:
        return bool(self.contains(np.asarray(p)[None])[0])

    def signed_distance(self, p) -> float:
        """Distance to the boundary; positive inside, negative outside."""
        p = np.asarray(p, dtype=np.float64)
        a = self.pts
        b = np.roll(self.pts, -1, axis=0)
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        proj = a + ab * t[:, None]
        d = float(np.min(np.linalg.norm(proj - p, axis=1)))
        return d if self.contains_point(p) else -d

    def bbox(self) -> tuple[float, float, float, float]:
        return (float(self.pts[:, 0].min()), float(self.pts[:, 1].min()),
                float(self.pts[:, 0].max()), float(self.pts[:, 1].max()))

    def area(self) -> float:
        x, y = self.pts[:, 0], self.pts[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


class Polyline:
    def __init__(self, pts: np.ndarray, name: str = ""):
        self.pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        self.name = name
        if len(self.pts) < 2:
            raise ValueError(f"polyline {name!r} needs >= 2 points")
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def project(self, p) -> tuple[float, float, int, float]:
        """Return (arc_length, distance, segment_index, t_on_segment) of the
        closest point on the polyline to ``p``."""
        p = np.asarray(p, dtype=np.float64)
        a = self.pts[:-1]
        ab = np.diff(self.pts, axis=0)
        denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
        t_raw = np.einsum("ij,ij->i", p - a, ab) / denom
        t = np.clip(t_raw, 0, 1)
        proj = a + ab * t[:, None]
        d = np.linalg.norm(proj - p, axis=1)
        i = int(np.argmin(d))
        return float(self.cum[i] + t[i] * self.seg_len[i]), float(d[i]), i, float(t_raw[i])

    def tangent_at(self, p) -> np.ndarray:
        _, _, i, _ = self.project(p)
        v = self.pts[i + 1] - self.pts[i]
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([1.0, 0.0])

    def side(self, p) -> int:
        """+1 / -1 depending on which side of the nearest segment ``p`` lies,
        0 if exactly on it."""
        _, _, i, _ = self.project(p)
        a, b = self.pts[i], self.pts[i + 1]
        c = float(_cross2(b - a, np.asarray(p, dtype=np.float64) - a))
        return 0 if abs(c) < 1e-9 else (1 if c > 0 else -1)

    def within_extent(self, p, tol: float = 0.0) -> bool:
        """True if ``p`` projects onto the polyline (not beyond its ends)."""
        _, _, i0, t0 = self.project(p)
        if i0 == 0 and t0 < 0:
            return bool(-t0 * self.seg_len[0] <= tol)
        last = len(self.seg_len) - 1
        if i0 == last and t0 > 1:
            return bool((t0 - 1) * self.seg_len[last] <= tol)
        return True

    def side_many(self, pts, tol=0.0) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised ``side`` and ``within_extent`` for an (N, 2) array.

        Returns (side in {-1, 0, 1}, within_extent bool); ``tol`` may be a
        scalar or an (N,) array (pixels allowed beyond the polyline ends)."""
        P = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if len(P) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
        a = self.pts[:-1]
        ab = np.diff(self.pts, axis=0)
        denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
        ap = P[:, None, :] - a[None]
        t_raw = np.einsum("nsk,sk->ns", ap, ab) / denom
        proj = a[None] + ab[None] * np.clip(t_raw, 0, 1)[..., None]
        d = np.linalg.norm(proj - P[:, None, :], axis=2)
        i = np.argmin(d, axis=1)
        tr = t_raw[np.arange(len(P)), i]
        c = _cross2(ab[i], P - a[i])
        side = np.where(np.abs(c) < 1e-9, 0, np.sign(c)).astype(np.int64)
        last = len(self.seg_len) - 1
        over = np.where((i == 0) & (tr < 0), -tr * self.seg_len[0],
                        np.where((i == last) & (tr > 1), (tr - 1) * self.seg_len[last], 0.0))
        return side, over <= tol

    def crossed_by(self, p0, p1) -> bool:
        return any(segments_intersect(p0, p1, self.pts[i], self.pts[i + 1]) for i in range(len(self.pts) - 1))


# --------------------------------------------------------------------------
# scene objects
# --------------------------------------------------------------------------
@dataclass
class Lane:
    id: str
    group: str
    polygon: Polygon
    path: Polyline

    def direction_at(self, p) -> np.ndarray:
        return self.path.tangent_at(p)

    def heading_at(self, p) -> float:
        v = self.direction_at(p)
        return math.atan2(v[1], v[0])


@dataclass
class StopLine:
    id: str
    line: Polyline
    lanes: list[str]
    signal: str | None
    approach: np.ndarray            # unit vector of legal travel direction across the line
    right_turn_on_red: bool = False

    def side(self, p) -> int:
        """-1 before the line (approach side), +1 past it."""
        a, b = self.line.pts[0], self.line.pts[-1]
        normal = np.array([-(b - a)[1], (b - a)[0]])
        if float(normal @ self.approach) < 0:
            normal = -normal
        c = float(normal @ (np.asarray(p, dtype=np.float64) - a))
        return 1 if c > 0 else -1

    def distance_past(self, p) -> float:
        """Signed distance (px) of ``p`` along the approach normal (+ = past)."""
        a, b = self.line.pts[0], self.line.pts[-1]
        normal = np.array([-(b - a)[1], (b - a)[0]], dtype=np.float64)
        normal /= max(np.linalg.norm(normal), 1e-9)
        if float(normal @ self.approach) < 0:
            normal = -normal
        return float(normal @ (np.asarray(p, dtype=np.float64) - a))


@dataclass
class SignalSpec:
    id: str
    roi: tuple[int, int, int, int]       # pixel box x0, y0, x1, y1
    layout: str
    lamps: list[str]


@dataclass
class ProhibitedTurn:
    id: str
    label: str
    from_zone: Polygon
    to_zone: Polygon
    max_duration: float


@dataclass
class Geometry:
    width: int
    height: int
    calibrated: bool = False
    carriageway: list[Polygon] = field(default_factory=list)
    exclusion_zones: list[Polygon] = field(default_factory=list)
    sidewalks: list[Polygon] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    crossings: list[Polygon] = field(default_factory=list)
    intersection: Polygon | None = None
    signals: list[SignalSpec] = field(default_factory=list)
    solid_lines: list[Polyline] = field(default_factory=list)
    prohibited_turns: list[ProhibitedTurn] = field(default_factory=list)
    u_turn_prohibited: bool = False
    u_turn_zones: list[Polygon] = field(default_factory=list)
    obstacle_regions: list[Polygon] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ---------------- queries ----------------
    @property
    def has_carriageway(self) -> bool:
        return bool(self.carriageway)

    def on_carriageway(self, pts) -> np.ndarray | None:
        """Boolean mask, or None if no carriageway is configured."""
        if not self.carriageway:
            return None
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        inside = np.zeros(len(p), dtype=bool)
        for poly in self.carriageway:
            inside |= poly.contains(p)
        return inside

    def carriageway_depth(self, p) -> float:
        """Signed distance of ``p`` to the carriageway boundary (+ inside)."""
        if not self.carriageway:
            return -math.inf
        return max(poly.signed_distance(p) for poly in self.carriageway)

    def in_any(self, polys: list[Polygon], pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        out = np.zeros(len(p), dtype=bool)
        for poly in polys:
            out |= poly.contains(p)
        return out

    def lane_of(self, p) -> Lane | None:
        """Lane containing ``p``; ties broken by distance to the lane path."""
        best, best_d = None, math.inf
        for lane in self.lanes:
            if lane.polygon.contains_point(p):
                d = lane.path.project(p)[1]
                if d < best_d:
                    best, best_d = lane, d
        return best

    def lane_groups(self) -> dict[str, list[Lane]]:
        groups: dict[str, list[Lane]] = {}
        for lane in self.lanes:
            groups.setdefault(lane.group, []).append(lane)
        return dict(sorted(groups.items()))

    def signal(self, sid: str | None) -> SignalSpec | None:
        for s in self.signals:
            if s.id == sid:
                return s
        return None

    def summary(self) -> dict:
        return {
            "calibrated": self.calibrated,
            "carriageway": len(self.carriageway),
            "lanes": len(self.lanes),
            "stop_lines": len(self.stop_lines),
            "crossings": len(self.crossings),
            "intersection": self.intersection is not None,
            "signals": len(self.signals),
            "solid_lines": len(self.solid_lines),
            "prohibited_turns": len(self.prohibited_turns),
            "u_turn_prohibited": self.u_turn_prohibited,
            "u_turn_zones": len(self.u_turn_zones),
        }


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _check_norm(points, what: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{what}: expected a list of [x, y] points")
    if np.any(arr < -0.05) or np.any(arr > 1.05):
        raise ValueError(f"{what}: coordinates must be normalised to [0, 1]")
    return arr


def load_geometry_dict(path: str | Path | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        log.warning("geometry file %s not found - running uncalibrated", p)
        return {}
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return data


def build_geometry(data: dict, width: int, height: int) -> Geometry:
    """Convert a normalised geometry mapping to pixel geometry.

    Invalid entries are skipped with a warning; they never raise, so a typo in
    the YAML degrades gracefully to "feature not configured".
    """
    g = Geometry(width=width, height=height, calibrated=bool(data.get("calibrated", False)))

    def px(points, what):
        return to_px(_check_norm(points, what), width, height)

    def safe(what, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - config errors must not crash
            msg = f"geometry: skipping {what}: {exc}"
            g.warnings.append(msg)
            log.warning(msg)
            return None

    for key, target in (("carriageway", g.carriageway), ("exclusion_zones", g.exclusion_zones),
                        ("sidewalks", g.sidewalks), ("u_turn_prohibited_zones", g.u_turn_zones),
                        ("obstacle_regions", g.obstacle_regions)):
        for i, poly in enumerate(data.get(key) or []):
            item = safe(f"{key}[{i}]", lambda poly=poly, i=i, key=key: Polygon(px(poly, f"{key}[{i}]"), f"{key}{i}"))
            if item is not None:
                target.append(item)

    for i, c in enumerate(data.get("crossings") or []):
        def mk_cross(c=c, i=i):
            pts = c["polygon"] if isinstance(c, dict) else c
            name = c.get("id", f"crossing{i}") if isinstance(c, dict) else f"crossing{i}"
            return Polygon(px(pts, f"crossings[{i}]"), name)
        item = safe(f"crossings[{i}]", mk_cross)
        if item is not None:
            g.crossings.append(item)

    if data.get("intersection"):
        g.intersection = safe("intersection", lambda: Polygon(px(data["intersection"], "intersection"), "intersection"))

    for i, ln in enumerate(data.get("lanes") or []):
        def mk_lane(ln=ln, i=i):
            lid = str(ln.get("id", f"lane{i}"))
            return Lane(lid, str(ln.get("group", lid)), Polygon(px(ln["polygon"], f"lanes[{i}].polygon"), lid),
                        Polyline(px(ln["direction"], f"lanes[{i}].direction"), lid))
        item = safe(f"lanes[{i}]", mk_lane)
        if item is not None:
            g.lanes.append(item)
    lanes_by_id = {lane.id: lane for lane in g.lanes}

    for i, s in enumerate(data.get("signals") or []):
        def mk_sig(s=s, i=i):
            roi = np.asarray(s["roi"], dtype=np.float64).reshape(4)
            _check_norm(roi.reshape(2, 2), f"signals[{i}].roi")
            x0, x1 = sorted((roi[0] * width, roi[2] * width))
            y0, y1 = sorted((roi[1] * height, roi[3] * height))
            if x1 - x0 < 2 or y1 - y0 < 2:
                raise ValueError("signal roi too small")
            layout = str(s.get("layout", "vertical"))
            if layout not in ("vertical", "horizontal"):
                raise ValueError("layout must be vertical or horizontal")
            lamps = [str(x) for x in s.get("lamps", ["red", "amber", "green"])]
            return SignalSpec(str(s.get("id", f"signal{i}")), (int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))),
                              layout, lamps)
        item = safe(f"signals[{i}]", mk_sig)
        if item is not None:
            g.signals.append(item)

    for i, sl in enumerate(data.get("stop_lines") or []):
        def mk_sl(sl=sl, i=i):
            line = Polyline(px(sl["line"], f"stop_lines[{i}].line"), str(sl.get("id", f"stop{i}")))
            lanes = [str(x) for x in sl.get("lanes") or []]
            if sl.get("approach"):
                ap = px(sl["approach"], f"stop_lines[{i}].approach")
                vec = ap[-1] - ap[0]
            else:
                mid = (line.pts[0] + line.pts[-1]) / 2
                vecs = [lanes_by_id[lid].direction_at(mid) for lid in lanes if lid in lanes_by_id]
                if not vecs:
                    raise ValueError("needs `approach` or `lanes` with directions")
                vec = np.mean(vecs, axis=0)
            n = np.linalg.norm(vec)
            if n < 1e-9:
                raise ValueError("degenerate approach direction")
            return StopLine(line.name, line, lanes, sl.get("signal"), vec / n, bool(sl.get("right_turn_on_red", False)))
        item = safe(f"stop_lines[{i}]", mk_sl)
        if item is not None:
            g.stop_lines.append(item)

    for i, sl in enumerate(data.get("solid_lines") or []):
        def mk_solid(sl=sl, i=i):
            pts = sl["points"] if isinstance(sl, dict) else sl
            name = sl.get("id", f"solid{i}") if isinstance(sl, dict) else f"solid{i}"
            return Polyline(px(pts, f"solid_lines[{i}]"), str(name))
        item = safe(f"solid_lines[{i}]", mk_solid)
        if item is not None:
            g.solid_lines.append(item)

    for i, pt in enumerate(data.get("prohibited_turns") or []):
        def mk_turn(pt=pt, i=i):
            label = str(pt.get("label", "illegal_turn"))
            if label not in ("illegal_turn", "illegal_u_turn"):
                raise ValueError("label must be illegal_turn or illegal_u_turn")
            tid = str(pt.get("id", f"turn{i}"))
            return ProhibitedTurn(tid, label, Polygon(px(pt["from"], f"prohibited_turns[{i}].from"), tid + "_from"),
                                  Polygon(px(pt["to"], f"prohibited_turns[{i}].to"), tid + "_to"),
                                  float(pt.get("max_duration_sec", 12.0)))
        item = safe(f"prohibited_turns[{i}]", mk_turn)
        if item is not None:
            g.prohibited_turns.append(item)

    g.u_turn_prohibited = bool(data.get("u_turn_prohibited", False))
    return g


def load_geometry(path: str | Path | None, width: int, height: int) -> Geometry:
    try:
        return build_geometry(load_geometry_dict(path), width, height)
    except Exception as exc:  # noqa: BLE001
        log.warning("geometry %s unusable (%s) - running uncalibrated", path, exc)
        return Geometry(width=width, height=height)
