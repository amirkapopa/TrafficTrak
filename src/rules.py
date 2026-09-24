"""Transparent, class-specific event rules.

Every rule consumes smoothed track series (``features.TrackSeries``), the
scene geometry, the learned flow field and signal timelines, and returns
``RawEvent`` segments.  Rules whose prerequisites (e.g. a calibrated stop
line) are missing return nothing - they never guess.  Each rule is wrapped so
that a failure in one rule is logged and cannot affect the others.

Timing convention: a sample at grid time t covers [t - dt/2, t + dt/2), so a
run of samples i0..i1 becomes the segment [t(i0) - dt/2, t(i1) + dt/2].
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .config import class_cfg, class_enabled
from .features import PairFeatures, TrackSeries, candidate_pairs, drop_short, fill_gaps, pair_features, runs, sustained_onset
from .flow import FlowField, nearest_group
from .geometry import Geometry, angle_diff
from .monitors import TimedFlag
from .segments import RawEvent
from .signal_state import SignalTimeline

log = logging.getLogger("traffictrak.rules")

VEHICLE_GROUPS = ("vehicle", "two_wheeler")


@dataclass
class SceneContext:
    cfg: dict
    geometry: Geometry
    flow: FlowField
    series: list[TrackSeries]
    width: int
    height: int
    dt: float
    duration: float
    signals: dict[str, SignalTimeline] = field(default_factory=dict)
    obstacle_flags: TimedFlag | None = None
    obstacle_every: float = 0.5
    fire_flags: TimedFlag | None = None
    fire_every: float = 0.5
    direction_groups: list[float] = field(default_factory=list)

    # ------------------------------------------------------------ helpers
    def n(self, sec: float) -> int:
        return max(1, int(round(sec / self.dt)))

    def span(self, s: TrackSeries, i0: int, i1: int) -> tuple[float, float]:
        return s.time(i0) - self.dt / 2, s.time(i1) + self.dt / 2

    @property
    def feat(self) -> dict:
        return self.cfg.get("features", {})

    def of_classes(self, classes) -> list[TrackSeries]:
        cl = set(classes)
        return [s for s in self.series if s.cls in cl]

    def of_groups(self, groups) -> list[TrackSeries]:
        gs = set(groups)
        return [s for s in self.series if s.group in gs]

    def learned_road_available(self) -> bool:
        return self.flow is not None and float(self.flow.occ.max(initial=0.0)) >= \
            float(self.cfg.get("scene", {}).get("carriageway_min_tracks", 3))

    def on_road(self, p, tid: int | None = None, allow_learned: bool = True) -> bool:
        """Is pixel point ``p`` on the carriageway (manual geometry first)?"""
        g = self.geometry
        pts = np.asarray(p, dtype=np.float64)[None]
        if g.exclusion_zones and g.in_any(g.exclusion_zones, pts)[0]:
            return False
        if g.sidewalks and g.in_any(g.sidewalks, pts)[0]:
            return False
        manual = g.on_carriageway(pts)
        if manual is not None:
            return bool(manual[0])
        if allow_learned and self.learned_road_available():
            return self.flow.carriageway(float(p[0]), float(p[1]), self.width, self.height,
                                         float(self.cfg.get("scene", {}).get("carriageway_min_tracks", 3)), exclude=tid)
        return False

    def flow_direction(self, p, tid: int | None, min_support: float, min_conc: float) -> float | None:
        ang, conc, sup = self.flow.dominant(float(p[0]), float(p[1]), self.width, self.height, exclude=tid)
        if sup >= min_support and conc >= min_conc:
            return ang
        return None


def _by_frame(series: list[TrackSeries]) -> dict[int, list[tuple[TrackSeries, int]]]:
    idx: dict[int, list[tuple[TrackSeries, int]]] = {}
    for s in series:
        for i in range(s.n):
            idx.setdefault(s.k0 + i, []).append((s, i))
    return idx


def _box_overlap_frac(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection area / area(a) for row-aligned xyxy arrays."""
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-6)
    return inter / area


def rider_or_occupant_mask(ctx: SceneContext, person: TrackSeries, thr: float) -> np.ndarray:
    """True where the person is riding a two-wheeler or inside a vehicle box."""
    mask = np.zeros(person.n, dtype=bool)
    for o in ctx.of_groups(("two_wheeler", "vehicle")):
        k0, k1 = max(o.k0, person.k0), min(o.k1, person.k1)
        if k1 < k0:
            continue
        ks = np.arange(k0, k1 + 1)
        ip, io = ks - person.k0, ks - o.k0
        frac = _box_overlap_frac(person.box[ip], o.box[io])
        lim = thr if o.group == "two_wheeler" else 0.6
        mask[ip] |= frac >= lim
    return mask


# ======================================================================
# stopped_vehicle
# ======================================================================
def rule_stopped_vehicle(ctx: SceneContext) -> list[RawEvent]:
    c = class_cfg(ctx.cfg, "stopped_vehicle")
    stationary = float(ctx.feat.get("stationary_speed", 0.12))
    moving = float(ctx.feat.get("moving_speed", 0.4))
    crawl = float(ctx.feat.get("crawl_speed", 1.0))
    vehicles = ctx.of_classes(c.get("vehicle_classes", ["car", "bus", "truck", "motorcycle"]))
    others_all = ctx.of_groups(VEHICLE_GROUPS)
    out = []
    for s in vehicles:
        w = ctx.n(float(c.get("displacement_window_sec", 1.0)))
        g = s.ground()
        ar = np.arange(s.n)
        disp = np.linalg.norm(g[np.clip(ar + w, 0, s.n - 1)] - g[np.clip(ar - w, 0, s.n - 1)], axis=1) / s.scale
        stat = (s.speed_n < stationary) & (disp < float(c.get("max_displacement", 0.25)))
        stat = fill_gaps(stat, ctx.n(float(c.get("gap_fill_sec", 1.0))))
        for i0, i1 in runs(stat):
            dur = (i1 - i0 + 1) * ctx.dt
            if dur < float(c.get("min_duration_sec", 10.0)):
                continue
            p = np.median(g[i0:i1 + 1], axis=0)
            if not ctx.on_road(p, s.tid):
                continue
            sc = float(np.median(s.scale[i0:i1 + 1]))
            # direction the vehicle was travelling
            moved = np.where(s.speed_n[:i0] >= moving)[0]
            direction = float(s.heading[moved[-1]]) if moved.size else ctx.flow_direction(p, s.tid, 3, 0.6)
            radius = float(c.get("queue_radius", 6.0)) * sc
            passers: set[int] = set()
            queue_samples = neighbour_samples = 0
            ks = np.arange(s.k0 + i0, s.k0 + i1 + 1, ctx.n(1.0))
            for k in ks:
                queue_here = neighbour_here = False
                for o in others_all:
                    if o.tid == s.tid:
                        continue
                    j = o.local(int(k))
                    if j < 0:
                        continue
                    vec = np.array([o.gx[j] - p[0], o.gy[j] - p[1]])
                    d = float(np.hypot(*vec))
                    if d > radius:
                        continue
                    neighbour_here = True
                    if o.speed_n[j] >= moving:
                        if direction is None or angle_diff(o.heading[j], direction) < math.radians(60):
                            passers.add(o.tid)
                    elif o.speed_n[j] < crawl:
                        if direction is None:
                            queue_here = True
                        else:
                            along = abs(math.cos(math.atan2(vec[1], vec[0]) - direction))
                            if along >= math.cos(math.radians(35)):
                                queue_here = True
                queue_samples += queue_here
                neighbour_samples += neighbour_here
            n_samples = max(len(ks), 1)
            frac_queue = queue_samples / n_samples
            if _waiting_at_red(ctx, s, i0, i1, p, sc, c):
                continue
            if frac_queue >= float(c.get("queue_frac", 0.5)):
                continue
            emit = len(passers) >= int(c.get("passers_min", 2))
            if not emit and neighbour_samples == 0 and dur >= float(c.get("isolated_min_duration_sec", 45.0)):
                emit = True
            if emit:
                t0, t1 = ctx.span(s, i0, i1)
                out.append(RawEvent("stopped_vehicle", t0, t1, 0.8, (s.tid,),
                                    f"stationary {dur:.1f}s, {len(passers)} passers, queue {frac_queue:.2f}"))
    return out


def _waiting_at_red(ctx: SceneContext, s: TrackSeries, i0: int, i1: int, p, sc: float, c: dict) -> bool:
    """Normal signal queue: stopped near a stop line while its signal is red."""
    for sl in ctx.geometry.stop_lines:
        tl = ctx.signals.get(str(sl.signal))
        if tl is None:
            continue
        d = abs(sl.distance_past(p))
        if d > float(c.get("stop_line_radius", 3.0)) * sc:
            continue
        ts = np.arange(s.time(i0), s.time(i1), 1.0)
        if len(ts) and np.mean([tl.state_at(t) == "red" for t in ts]) >= 0.5:
            return True
    return False


# ======================================================================
# wrong_way
# ======================================================================
def rule_wrong_way(ctx: SceneContext) -> list[RawEvent]:
    c = class_cfg(ctx.cfg, "wrong_way")
    g = ctx.geometry
    moving = float(ctx.feat.get("moving_speed", 0.4))
    min_angle = math.radians(float(c.get("min_angle_deg", 135)))
    use_lanes = bool(g.lanes)
    if not use_lanes and float(ctx.flow.support.sum()) <= 0:
        return []
    out = []
    for s in ctx.of_classes(c.get("vehicle_classes", ["car", "bus", "truck", "motorcycle"])):
        mov = s.speed_n >= moving
        opp = np.zeros(s.n, dtype=bool)
        known = np.zeros(s.n, dtype=bool)
        lane_ids: list[str | None] = [None] * s.n
        for i in np.where(mov)[0]:
            p = (s.gx[i], s.gy[i])
            if use_lanes:
                lane = g.lane_of(p)
                if lane is None:
                    continue
                lane_ids[i] = lane.id
                ref = lane.heading_at(p)
            else:
                ref = ctx.flow_direction(p, s.tid, float(c.get("auto_min_support", 6)),
                                         float(c.get("auto_min_concentration", 0.85)))
                if ref is None:
                    continue
            known[i] = True
            opp[i] = angle_diff(s.heading[i], ref) >= min_angle
        cand = opp | (mov & ~known)
        cand = fill_gaps(cand, ctx.n(0.5))
        for i0, i1 in runs(cand):
            seg_opp = opp[i0:i1 + 1]
            if seg_opp.sum() < ctx.n(float(c.get("min_duration_sec", 1.5))):
                continue
            seg_known = known[i0:i1 + 1]
            if seg_known.sum() == 0 or seg_opp.sum() / max(seg_known.sum(), 1) < float(c.get("auto_min_qualified_frac", 0.6)):
                continue
            # trim to first/last opposing sample
            idx = np.where(opp[i0:i1 + 1])[0]
            a, b = i0 + int(idx[0]), i0 + int(idx[-1])
            if s.path_length(a, b) < float(c.get("min_path", 3.0)):
                continue
            if use_lanes and lane_ids[a] is not None:
                # walk back to the moment the vehicle entered this lane
                lane = next(ln for ln in g.lanes if ln.id == lane_ids[a])
                while a > 0 and lane.polygon.contains_point((s.gx[a - 1], s.gy[a - 1])):
                    a -= 1
            t0, t1 = ctx.span(s, a, b)
            out.append(RawEvent("wrong_way", t0, t1, 0.8, (s.tid,), "lanes" if use_lanes else "learned flow"))
    return out


# ======================================================================
# congestion
# ======================================================================
def rule_congestion(ctx: SceneContext) -> list[RawEvent]:
    c = class_cfg(ctx.cfg, "congestion")
    g = ctx.geometry
    crawl = float(ctx.feat.get("crawl_speed", 1.0))
    bin_sec = float(c.get("bin_sec", 1.0))
    nb = int(math.ceil(ctx.duration / bin_sec)) if ctx.duration > 0 else 0
    if nb == 0:
        return []
    if g.lanes:
        group_names = list(g.lane_groups().keys())
    else:
        group_names = [f"dir{i}" for i in range(len(ctx.direction_groups))]
    if not group_names:
        return []
    G = len(group_names)
    count = np.zeros((G, nb))
    slow = np.zeros((G, nb))
    speeds: list[list[list[float]]] = [[[] for _ in range(nb)] for _ in range(G)]
    lanes_slow: dict[tuple[int, int], set] = {}
    for s in ctx.of_groups(("vehicle",)):
        b_of = np.minimum((s.t / bin_sec).astype(np.int64), nb - 1)
        for b in np.unique(b_of):
            sel = np.where(b_of == b)[0]
            i = int(sel[len(sel) // 2])
            p = (float(s.gx[i]), float(s.gy[i]))
            if g.carriageway:
                if not ctx.on_road(p, s.tid, allow_learned=False):
                    continue
            elif ctx.learned_road_available() and not ctx.on_road(p, s.tid):
                continue
            lane_id = None
            if g.lanes:
                lane = g.lane_of(p)
                if lane is None:
                    continue
                gi = group_names.index(lane.group)
                lane_id = lane.id
            else:
                ang = ctx.flow_direction(p, None, 3, 0.6)
                if ang is None:
                    if s.speed_n[i] < float(ctx.feat.get("moving_speed", 0.4)):
                        continue
                    ang = float(s.heading[i])
                gi = nearest_group(ang, ctx.direction_groups)
                if gi < 0:
                    continue
            v = float(np.median(s.speed_n[sel]))
            count[gi, b] += 1
            speeds[gi][b].append(v)
            if v < crawl:
                slow[gi, b] += 1
                if lane_id is not None:
                    lanes_slow.setdefault((gi, int(b)), set()).add(lane_id)
    out = []
    lane_groups = g.lane_groups()
    for gi, name in enumerate(group_names):
        nz = count[gi][count[gi] > 0]
        if nz.size == 0:
            continue
        cap = float(np.percentile(nz, 95))
        need = max(float(c.get("min_vehicles", 6)), float(c.get("capacity_frac", 0.5)) * cap)
        med = np.array([np.median(x) if x else np.inf for x in speeds[gi]])
        jam = (count[gi] >= need) & (slow[gi] >= float(c.get("slow_frac", 0.7)) * np.maximum(count[gi], 1)) & (med <= crawl)
        if g.lanes:
            all_lanes = {ln.id for ln in lane_groups.get(name, [])}
            for b in np.where(jam)[0]:
                if lanes_slow.get((gi, int(b)), set()) != all_lanes:
                    jam[b] = False
        jam = fill_gaps(jam, max(1, int(round(float(c.get("gap_fill_sec", 5.0)) / bin_sec))))
        for b0, b1 in runs(jam):
            if (b1 - b0 + 1) * bin_sec < float(c.get("min_duration_sec", 30.0)):
                continue
            out.append(RawEvent("congestion", b0 * bin_sec, (b1 + 1) * bin_sec, 0.7, (), f"group {name}"))
    return out


# ======================================================================
# jaywalking & failure_to_yield
# ======================================================================
def _crossing_mask(g: Geometry, pts: np.ndarray, buffer_px: np.ndarray | None = None) -> np.ndarray:
    inside = g.in_any(g.crossings, pts)
    if buffer_px is not None and len(pts):
        for j in np.where(~inside)[0]:
            if any(poly.signed_distance(pts[j]) >= -buffer_px[j] for poly in g.crossings):
                inside[j] = True
    return inside


def rule_jaywalking(ctx: SceneContext) -> list[RawEvent]:
    c = class_cfg(ctx.cfg, "jaywalking")
    g = ctx.geometry
    if not g.carriageway and not c.get("use_learned_carriageway", False):
        return []
    out = []
    for s in ctx.of_groups(("person",)):
        pts = s.ground()
        rider = rider_or_occupant_mask(ctx, s, float(c.get("rider_overlap", 0.3)))
        margin = float(c.get("road_margin", 0.15)) * s.scale
        if g.carriageway:
            on = g.on_carriageway(pts)
            for i in np.where(on)[0]:
                if g.carriageway_depth(pts[i]) < margin[i]:
                    on[i] = False
        else:
            on = np.array([ctx.on_road(p, s.tid) for p in pts], dtype=bool)
        if g.crossings:
            on &= ~_crossing_mask(g, pts, 0.3 * s.scale)
        if g.sidewalks or g.exclusion_zones:
            on &= ~g.in_any(g.sidewalks + g.exclusion_zones, pts)
        on &= ~rider
        on = fill_gaps(on, ctx.n(0.5))
        on = drop_short(on, ctx.n(float(c.get("min_duration_sec", 1.0))))
        for i0, i1 in runs(on):
            t0, t1 = ctx.span(s, i0, i1)
            out.append(RawEvent("jaywalking", t0, t1, 0.7, (s.tid,)))
    return out


def rule_failure_to_yield(ctx: SceneContext) -> list[RawEvent]:
    c = class_cfg(ctx.cfg, "failure_to_yield")
    g = ctx.geometry
    if not g.crossings:
        return []
    persons = ctx.of_groups(("person",))
    vehicles = ctx.of_classes(("car", "bus", "truck", "motorcycle"))
    out = []
    for poly in g.crossings:
        # pedestrians on / entering the crossing, indexed by grid step
        ped_at: dict[int, list[tuple[float, float]]] = {}
        centre = poly.pts.mean(axis=0)
        for s in persons:
            rider = rider_or_occupant_mask(ctx, s, 0.3)
            pts = s.ground()
            inside = poly.contains(pts)
            buf = float(c.get("entering_buffer", 0.6)) * s.scale
            for i in range(s.n):
                if rider[i]:
                    continue
                ok = bool(inside[i])
                if not ok and poly.signed_distance(pts[i]) >= -buf[i]:
                    toward = (centre - pts[i]) @ np.array([s.vx[i], s.vy[i]])
                    ok = toward > 0 and s.speed_n[i] >= 0.2
                if ok:
                    ped_at.setdefault(s.k0 + i, []).append((float(pts[i, 0]), float(pts[i, 1])))
        if not ped_at:
            continue
        for v in vehicles:
            inset = 0.15 * v.w
            probe = [np.stack([v.box[:, 0] + inset, v.box[:, 3]], 1), v.ground(), np.stack([v.box[:, 2] - inset, v.box[:, 3]], 1)]
            inside = np.zeros(v.n, dtype=bool)
            for pts in probe:
                inside |= poly.contains(pts)
            inside = fill_gaps(inside, ctx.n(0.3))
            for i0, i1 in runs(inside):
                if float(np.mean(v.speed_n[i0:i1 + 1])) < float(c.get("min_vehicle_speed", 0.4)):
                    continue
                conflict = False
                for i in range(i0, i1 + 1):
                    for (px, py) in ped_at.get(v.k0 + i, []):
                        if math.hypot(px - v.gx[i], py - v.gy[i]) <= float(c.get("max_ped_distance", 8.0)) * v.scale[i]:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    t0, t1 = ctx.span(v, i0, i1)
                    out.append(RawEvent("failure_to_yield", t0, t1, 0.7, (v.tid,), poly.name))
    return out


# ======================================================================
# red_light & stop_line
# ======================================================================
def _front_points(v: TrackSeries, approach: np.ndarray, frac: float) -> np.ndarray:
    fx = v.gx + approach[0] * v.w / 2
    fy = v.gy + min(0.0, float(approach[1])) * frac * v.h
    return np.stack([fx, fy], 1)


def rule_signal_violations(ctx: SceneContext) -> list[RawEvent]:
    g = ctx.geometry
    if not g.stop_lines or not ctx.signals:
        return []
    c_red = class_cfg(ctx.cfg, "red_light")
    c_stop = class_cfg(ctx.cfg, "stop_line")
    red_on, stop_on = class_enabled(ctx.cfg, "red_light"), class_enabled(ctx.cfg, "stop_line")
    min_rel = float(ctx.cfg.get("signal", {}).get("min_reliability", 0.6))
    stationary = float(ctx.feat.get("stationary_speed", 0.12))
    moving = float(ctx.feat.get("moving_speed", 0.4))
    out = []
    lanes_by_id = {ln.id: ln for ln in g.lanes}
    for sl in g.stop_lines:
        tl = ctx.signals.get(str(sl.signal))
        if tl is None or tl.reliability < min_rel:
            continue
        a = sl.approach
        for v in ctx.of_classes(("car", "bus", "truck", "motorcycle")):
            front = _front_points(v, a, float(c_red.get("front_offset_frac", 0.5)))
            past = np.array([sl.distance_past(p) for p in front])
            side, within = sl.line.side_many(front, 0.5 * v.scale)
            along = (v.vx * a[0] + v.vy * a[1]) > 0
            crosses = np.where((past[:-1] < 0) & (past[1:] >= 0) & within[1:] & along[1:])[0] + 1
            for ic in crosses.tolist():
                if sl.lanes:
                    back = max(0, ic - ctx.n(2.0))
                    in_lane = any(lanes_by_id[lid].polygon.contains(v.ground()[back:ic + 1]).any()
                                  for lid in sl.lanes if lid in lanes_by_id)
                    if not in_lane:
                        continue
                t_cross = v.time(ic)
                ground_past = np.array([sl.distance_past(p) for p in v.ground()])
                if g.intersection is not None:
                    entered = np.where(g.intersection.contains(v.ground()) & (np.arange(v.n) >= ic))[0]
                else:
                    entered = np.where((ground_past >= float(c_stop.get("max_past_line", 1.5)) * v.scale)
                                       & (np.arange(v.n) >= ic))[0]
                i_enter = int(entered[0]) if entered.size else -1
                i_stop = sustained_onset(v.speed_n < stationary, ic, ctx.n(float(c_stop.get("stop_hold_sec", 1.0))))
                if stop_on and i_stop >= 0 and (i_enter < 0 or i_stop < i_enter) \
                        and v.time(i_stop) - t_cross <= 5.0 and tl.state_at(v.time(i_stop)) == "red":
                    green = tl.next_change(v.time(i_stop), "green")
                    moves = np.where((v.speed_n >= moving) & (np.arange(v.n) > i_stop))[0]
                    ends = [v.t1 + ctx.dt / 2]
                    if green is not None:
                        ends.append(green)
                    if moves.size:
                        ends.append(v.time(int(moves[0])))
                    out.append(RawEvent("stop_line", v.time(i_stop) - ctx.dt / 2, min(ends), 0.7, (v.tid,), sl.id))
                    continue
                if not red_on or i_enter < 0 or v.time(i_enter) - t_cross > float(c_red.get("enter_within_sec", 3.0)):
                    continue
                grace = float(c_red.get("red_grace_sec", 0.4))
                if not tl.is_state_throughout("red", t_cross - grace, t_cross):
                    continue
                if sl.right_turn_on_red:
                    later = min(v.n - 1, ic + ctx.n(4.0))
                    d = np.array([v.vx[later], v.vy[later]])
                    if np.linalg.norm(d) > 0 and float(a[0] * d[1] - a[1] * d[0]) / np.linalg.norm(d) > math.sin(math.radians(45)):
                        continue  # right turn permitted on red
                if g.intersection is not None:
                    after = np.where(~g.intersection.contains(v.ground()) & (np.arange(v.n) > i_enter))[0]
                else:
                    after = np.where((ground_past >= float(c_red.get("exit_distance", 6.0)) * v.scale)
                                     & (np.arange(v.n) > ic))[0]
                i_end = int(after[0]) if after.size else v.n - 1
                out.append(RawEvent("red_light", t_cross - ctx.dt / 2, v.time(i_end) + ctx.dt / 2, 0.8, (v.tid,), sl.id))
    return out


# ======================================================================
# solid_line_crossing
# ======================================================================
def rule_solid_line(ctx: SceneContext) -> list[RawEvent]:
    c = class_cfg(ctx.cfg, "solid_line_crossing")
    g = ctx.geometry
    if not g.solid_lines:
        return []
    moving = float(ctx.feat.get("moving_speed", 0.4))
    max_n = ctx.n(float(c.get("max_duration_sec", 8)))
    confirm = ctx.n(float(c.get("confirm_sec", 0.5)))
    out = []
    for line in g.solid_lines:
        for v in ctx.of_classes(("car", "bus", "truck", "motorcycle")):
            inset = float(c.get("wheel_inset", 0.15)) * v.w
            L = np.stack([v.box[:, 0] + inset, v.box[:, 3]], 1)
            R = np.stack([v.box[:, 2] - inset, v.box[:, 3]], 1)
            C = v.ground()
            tol = 0.5 * v.scale
            sides = []
            ok = np.ones(v.n, dtype=bool)
            for P in (L, C, R):
                sd, within = line.side_many(P, tol)
                sides.append(sd)
                ok &= within
            S = np.stack(sides, 1)
            S[~ok] = 0
            known = np.where(np.all(S != 0, axis=1))[0]
            if known.size == 0:
                continue
            init = int(np.sign(S[known[0], 1]))
            i = int(known[0])
            while i < v.n:
                if not ok[i] or not np.any(S[i] == -init):
                    i += 1
                    continue
                start = i
                j = start
                while j < v.n and j - start <= max_n and not np.all(S[j] == -init):
                    j += 1
                if j >= v.n or j - start > max_n:
                    i = start + 1
                    # returned to the original side: reset scanning after this excursion
                    while i < v.n and np.any(S[i] == -init):
                        i += 1
                    continue
                stay = S[j:j + confirm, 1]
                if len(stay) >= min(confirm, v.n - j) and np.all(stay == -init) \
                        and float(np.max(v.speed_n[start:j + 1])) >= moving:
                    t0, t1 = ctx.span(v, start, j)
                    out.append(RawEvent("solid_line_crossing", t0, t1, 0.7, (v.tid,), line.name))
                    init = -init
                i = j + 1
    return out


# ======================================================================
# turns
# ======================================================================
def _turn_completion(v: TrackSeries, i_from: int, stable_rate: float, stable_n: int, cap: int) -> int:
    for i in range(i_from, min(v.n, i_from + cap)):
        seg = np.abs(v.heading_rate[i:i + stable_n])
        if len(seg) == stable_n and np.all(seg < stable_rate):
            return i
    return min(v.n - 1, i_from + cap - 1)


def rule_prohibited_turns(ctx: SceneContext) -> list[RawEvent]:
    g = ctx.geometry
    if not g.prohibited_turns:
        return []
    c = class_cfg(ctx.cfg, "illegal_turn")
    cu = class_cfg(ctx.cfg, "illegal_u_turn")
    onset = math.radians(float(c.get("onset_deg", 15)))
    stable = math.radians(float(c.get("stable_rate_deg", 12)))
    out = []
    for pt in g.prohibited_turns:
        if not class_enabled(ctx.cfg, pt.label):
            continue
        min_turn = math.radians(float(cu.get("min_turn_deg", 150)) if pt.label == "illegal_u_turn" else 30.0)
        for v in ctx.of_groups(VEHICLE_GROUPS):
            pts = v.ground()
            in_from, in_to = pt.from_zone.contains(pts), pt.to_zone.contains(pts)
            if not in_from.any() or not in_to.any():
                continue
            unwrapped = np.unwrap(v.heading_filled)
            for t0_idx, _ in runs(in_to):
                prev_from = np.where(in_from[:t0_idx])[0]
                if prev_from.size == 0:
                    continue
                i_from = int(prev_from[-1])
                if (t0_idx - i_from) * ctx.dt > pt.max_duration:
                    continue
                from_runs = [r for r in runs(in_from) if r[0] <= i_from <= r[1]]
                i_from_start = from_runs[0][0] if from_runs else i_from
                ref = float(np.mean(unwrapped[i_from_start:i_from_start + ctx.n(0.5)]))
                dev = np.abs(unwrapped - ref)
                look = min(v.n, t0_idx + ctx.n(3.0))
                if float(dev[i_from_start:look].max(initial=0.0)) < min_turn:
                    continue
                mask = dev >= onset
                on_idx = sustained_onset(mask, i_from_start, ctx.n(0.3))
                if on_idx < 0 or on_idx > t0_idx:
                    on_idx = i_from
                end_idx = _turn_completion(v, t0_idx, stable, ctx.n(0.5), ctx.n(5.0))
                t0, t1 = ctx.span(v, on_idx, max(end_idx, t0_idx))
                out.append(RawEvent(pt.label, t0, t1, 0.7, (v.tid,), pt.id))
    return out


def rule_u_turn(ctx: SceneContext) -> list[RawEvent]:
    g = ctx.geometry
    if not (g.u_turn_prohibited or g.u_turn_zones):
        return []
    c = class_cfg(ctx.cfg, "illegal_u_turn")
    moving = float(ctx.feat.get("moving_speed", 0.4))
    min_turn = math.radians(float(c.get("min_turn_deg", 150)))
    onset = math.radians(float(c.get("onset_deg", 20)))
    stable = math.radians(float(class_cfg(ctx.cfg, "illegal_turn").get("stable_rate_deg", 12)))
    win = ctx.n(float(c.get("max_duration_sec", 15)))
    out = []
    for v in ctx.of_groups(VEHICLE_GROUPS):
        if v.n < 3:
            continue
        u = np.unwrap(v.heading_filled)
        if float(u.max() - u.min()) < min_turn:
            continue
        jumps = np.r_[False, np.abs(np.diff(u)) > math.radians(60)]  # reversing, not turning
        j = 0
        floor = 0  # samples before this belong to an already reported manoeuvre
        while j < v.n:
            lo = max(floor, j - win)
            if jumps[lo + 1:j + 1].any():
                lo = lo + 1 + int(np.where(jumps[lo + 1:j + 1])[0][-1])
            seg = u[lo:j + 1]
            if len(seg) < 2 or float(seg.max() - seg.min()) < min_turn:
                j += 1
                continue
            i_ref = lo + int(np.argmax(np.abs(seg - u[j])))
            dev = np.abs(u - u[i_ref])
            on_idx = sustained_onset(dev >= onset, i_ref, ctx.n(0.3))
            if on_idx < 0:
                on_idx = i_ref
            end_idx = _turn_completion(v, j, stable, ctx.n(0.5), ctx.n(5.0))
            moving_frac = float(np.mean(v.speed_n[on_idx:end_idx + 1] >= moving)) if end_idx >= on_idx else 0.0
            p = (v.gx[on_idx], v.gy[on_idx])
            located = g.u_turn_prohibited or bool(g.in_any(g.u_turn_zones, np.array([p]))[0])
            if located and g.u_turn_prohibited and g.carriageway:
                located = ctx.on_road(p, v.tid, allow_learned=False)
            if moving_frac >= float(c.get("min_moving_frac", 0.6)) and located:
                t0, t1 = ctx.span(v, on_idx, end_idx)
                out.append(RawEvent("illegal_u_turn", t0, t1, 0.7, (v.tid,), "heading reversal"))
                floor = end_idx + 1
            j = max(end_idx, j) + 1
    return out


# ======================================================================
# accident & near_miss
# ======================================================================
def _speed_drop(v: TrackSeries, i: int, horizon: int, window: int) -> tuple[float, float]:
    """Largest drop in speed within ``window`` samples around index i.

    Returns (absolute drop, speed before)."""
    lo, hi = max(0, i - horizon), min(v.n - 1, i + horizon)
    best, before = 0.0, 0.0
    for a in range(lo, hi + 1):
        b = min(v.n - 1, a + window)
        drop = float(v.speed_n[a] - v.speed_n[a:b + 1].min())
        if drop > best:
            best, before = drop, float(v.speed_n[a])
    return best, before


def _velocity_jolt(v: TrackSeries, i: int, horizon: int, window: int) -> float:
    lo, hi = max(0, i - horizon), min(v.n - 1, i + horizon)
    best = 0.0
    for a in range(lo, hi + 1):
        b = min(v.n - 1, a + window)
        dv = math.hypot(v.vx[b] - v.vx[a], v.vy[b] - v.vy[a]) / max(float(v.scale[a]), 1.0)
        best = max(best, dv)
    return best


def _heading_jolt(v: TrackSeries, i: int, horizon: int, window: int, moving: float) -> float:
    lo, hi = max(0, i - horizon), min(v.n - 1, i + horizon)
    u = np.unwrap(v.heading_filled)
    best = 0.0
    for a in range(lo, hi + 1):
        b = min(v.n - 1, a + window)
        if v.speed_n[a] >= moving and v.speed_n[b] >= moving:
            best = max(best, abs(float(u[b] - u[a])))
    return best


def _fall(v: TrackSeries, i: int, horizon: int) -> bool:
    if v.group != "person":
        return False
    lo, hi = max(0, i - horizon), min(v.n - 1, i + horizon)
    ratio = v.h / np.maximum(v.w, 1.0)
    return bool(ratio[lo] > 0 and ratio[lo:hi + 1].min() < 0.6 * ratio[lo])


def _stop_time(v: TrackSeries, i: int, stationary: float, hold: int) -> int:
    k = sustained_onset(v.speed_n < stationary, i, hold)
    return k if k >= 0 else v.n - 1


def rule_accident_near_miss(ctx: SceneContext) -> list[RawEvent]:
    ca = class_cfg(ctx.cfg, "accident")
    cn = class_cfg(ctx.cfg, "near_miss")
    acc_on, nm_on = class_enabled(ctx.cfg, "accident"), class_enabled(ctx.cfg, "near_miss")
    band = float(ctx.feat.get("footprint_band", 0.35))
    stationary = float(ctx.feat.get("stationary_speed", 0.12))
    moving = float(ctx.feat.get("moving_speed", 0.4))
    crawl = float(ctx.feat.get("crawl_speed", 1.0))
    users = [s for s in ctx.series if s.group in ("vehicle", "two_wheeler", "person")
             and s.cls in set(ca.get("vehicle_classes", ["car", "bus", "truck", "motorcycle", "bicycle", "person"]))]
    pairs = candidate_pairs(users, ("vehicle", "two_wheeler"), ("vehicle", "two_wheeler", "person"), max_dist=3.0)
    accidents: list[RawEvent] = []
    near: list[RawEvent] = []
    acc_windows: dict[tuple[int, int], list[tuple[float, float]]] = {}
    h_imp = ctx.n(float(ca.get("impact_window_sec", 1.0)))
    w_half = ctx.n(0.5)
    for a, b in pairs:
        pf = pair_features(a, b, band, float(ca.get("depth_tolerance", 0.45)), float(ca.get("contact_gap", 0.1)))
        if pf is None:
            continue
        if acc_on:
            for j0, _ in runs(pf.contact):
                pre = pf.closing[max(0, j0 - ctx.n(0.7)):j0 + 1]
                if pre.size == 0 or float(pre.max()) < float(ca.get("min_closing_speed", 1.0)):
                    continue
                ia, ib = int(pf.ia[j0]), int(pf.ib[j0])
                decel = False
                for v, i in ((a, ia), (b, ib)):
                    drop, before = _speed_drop(v, i, h_imp, w_half)
                    if drop >= float(ca.get("decel_drop", 1.0)) and drop >= float(ca.get("decel_drop_frac", 0.5)) * max(before, 1e-6):
                        decel = True
                jolt_deg = math.radians(float(ca.get("heading_jolt_deg", 25)))
                second = any(_heading_jolt(v, i, h_imp, ctx.n(0.7), moving) >= jolt_deg or _fall(v, i, h_imp)
                             for v, i in ((a, ia), (b, ib)))
                # the slower party (struck object) receives a velocity jolt
                slower, si = (a, ia) if a.speed_n[max(ia - w_half, 0)] <= b.speed_n[max(ib - w_half, 0)] else (b, ib)
                second = second or _velocity_jolt(slower, si, w_half, w_half) >= float(ca.get("struck_velocity_jolt", 0.3))
                if not (decel and second):
                    continue
                post_n = ctx.n(float(ca.get("post_slow_sec", 2.0)))
                look = ctx.n(5.0)
                slow_at = [sustained_onset(v.speed_n < crawl, i, post_n) for v, i in ((a, ia), (b, ib))]
                post = all(0 <= k and k - i <= look for k, i in zip(slow_at, (ia, ib)))
                if not post:
                    continue
                hold = ctx.n(float(ca.get("stop_hold_sec", 1.0)))
                ends = [v.time(_stop_time(v, i, stationary, hold)) for v, i in ((a, ia), (b, ib))]
                t0 = pf.time(j0) - ctx.dt / 2
                t1 = min(max(ends) + ctx.dt / 2, t0 + float(ca.get("max_duration_sec", 60)))
                if t1 <= t0:
                    t1 = t0 + ctx.dt
                accidents.append(RawEvent("accident", t0, t1, 0.8, (a.tid, b.tid), "contact+impact"))
                acc_windows.setdefault((a.tid, b.tid), []).append((t0, t1))
                break
        if nm_on:
            near.extend(_near_miss_pair(ctx, pf, cn, acc_windows.get((a.tid, b.tid), []), moving))
    if acc_on and bool(ca.get("single_vehicle", False)):
        accidents.extend(_single_vehicle_accidents(ctx, ca, stationary))
    return accidents + near


def _parked_party(ctx: SceneContext, pf: PairFeatures, jc: int, moving: float, window: float) -> bool:
    """True if either party is a parked vehicle: (almost) never moving around the
    conflict and standing off the carriageway.  Without a ground-plane
    calibration, perspective makes a car stopping in its lane beside a kerb-side
    parked car look like a closing conflict."""
    w = ctx.n(window)
    for v, idx in ((pf.a, pf.ia), (pf.b, pf.ib)):
        if v.group == "person":
            continue
        i = int(idx[jc])
        lo, hi = max(0, i - w), min(v.n - 1, i + w)
        if float(np.mean(v.speed_n[lo:hi + 1] >= moving)) > 0.1:
            continue
        p = (float(np.median(v.gx[lo:hi + 1])), float(np.median(v.gy[lo:hi + 1])))
        road_known = ctx.geometry.has_carriageway or ctx.learned_road_available()
        if road_known and not ctx.on_road(p, v.tid):
            return True
    return False


def _near_miss_pair(ctx: SceneContext, pf: PairFeatures, c: dict, acc: list[tuple[float, float]], moving: float) -> list[RawEvent]:
    conflict = (pf.ttc < float(c.get("ttc_threshold", 1.0))) & (pf.closing >= float(c.get("min_closing_speed", 1.5)))
    if not conflict.any():
        return []
    out = []
    sustain = ctx.n(float(c.get("sustain_sec", 0.3)))
    before = ctx.n(float(c.get("evasive_window_before_sec", 2.5)))
    after = ctx.n(float(c.get("evasive_window_after_sec", 0.5)))
    min_v = float(c.get("min_speed_before", 1.0))
    last_end = -math.inf
    for jc, _ in runs(conflict):
        tc = pf.time(jc)
        if tc <= last_end:
            continue
        if any(s - float(c.get("accident_exclusion_sec", 3.0)) <= tc <= e + float(c.get("accident_exclusion_sec", 3.0)) for s, e in acc):
            continue
        if _parked_party(ctx, pf, jc, moving, float(c.get("parked_window_sec", 3.0))):
            continue
        onsets = []
        for v, idx in ((pf.a, pf.ia), (pf.b, pf.ib)):
            if v.group == "person":
                continue
            i = int(idx[jc])
            lo, hi = max(0, i - before), min(v.n - 1, i + after)
            fast = np.zeros(v.n, dtype=bool)
            fast[lo:hi + 1] = v.speed_n[lo:hi + 1] >= min_v * 0.5
            brake = fast & (v.accel_n <= -float(c.get("decel_threshold", 2.0)))
            swerve = fast & (np.abs(v.heading_rate) >= math.radians(float(c.get("swerve_rate_deg", 35))))
            for m in (brake, swerve):
                k = sustained_onset(m, lo, sustain)
                if 0 <= k <= hi and float(v.speed_n[max(k - ctx.n(0.5), 0):k + 1].max()) >= min_v:
                    onsets.append(v.time(k))
        if not onsets:
            continue
        t_on = min(onsets)
        # clearance: not closing and separated
        clear = (pf.closing <= float(c.get("clear_closing_speed", 0.2))) & (pf.gap >= float(c.get("clear_gap", 0.5)))
        k = sustained_onset(clear, jc, sustain)
        j_end = k if k >= 0 else pf.n - 1
        t_end = min(pf.time(j_end), t_on + float(c.get("max_duration_sec", 8)))
        # any physical contact in the window means it was not a *near* miss
        j_on = max(0, int(round((t_on / ctx.dt) - pf.k0)))
        if pf.contact[j_on:j_end + 1].any():
            continue
        if t_end > t_on:
            out.append(RawEvent("near_miss", t_on - ctx.dt / 2, t_end + ctx.dt / 2, 0.6, (pf.a.tid, pf.b.tid),
                                f"min ttc {float(np.min(pf.ttc[jc:j_end + 1])):.2f}s"))
            last_end = t_end
    return out


def _single_vehicle_accidents(ctx: SceneContext, c: dict, stationary: float) -> list[RawEvent]:
    out = []
    moving = float(ctx.feat.get("moving_speed", 0.4))
    vehicles = ctx.of_groups(VEHICLE_GROUPS)
    frames = _by_frame(vehicles)
    w = ctx.n(0.5)
    for v in vehicles:
        for i in range(1, v.n - w):
            if v.speed_n[i] < 2.0 or v.speed_n[i + w] >= stationary * 2:
                continue
            if _heading_jolt(v, i, w, w, moving) < math.radians(float(c.get("heading_jolt_deg", 25))):
                continue
            k = _stop_time(v, i, stationary, ctx.n(3.0))
            if k >= v.n - 1:
                continue
            near_other = any(o.tid != v.tid and math.hypot(o.gx[j] - v.gx[i + w], o.gy[j] - v.gy[i + w]) < 2 * v.scale[i + w]
                             for o, j in frames.get(v.k0 + i + w, []))
            if near_other:
                continue
            out.append(RawEvent("accident", v.time(i) - ctx.dt / 2, v.time(k) + ctx.dt / 2, 0.5, (v.tid,), "single vehicle"))
            break
    return out


# ======================================================================
# road_obstacle & fire_smoke
# ======================================================================
def rule_road_obstacle(ctx: SceneContext) -> list[RawEvent]:
    c = class_cfg(ctx.cfg, "road_obstacle")
    out = []
    if c.get("animals", True):
        for s in ctx.of_groups(("animal",)):
            on = np.array([ctx.on_road((s.gx[i], s.gy[i]), None) for i in range(s.n)], dtype=bool)
            on = fill_gaps(on, ctx.n(1.0))
            for i0, i1 in runs(on):
                if (i1 - i0 + 1) * ctx.dt >= float(c.get("animal_min_duration_sec", 1.0)):
                    t0, t1 = ctx.span(s, i0, i1)
                    out.append(RawEvent("road_obstacle", t0, t1, 0.7, (s.tid,), f"animal:{s.cls}"))
    f = ctx.obstacle_flags
    if f is not None and f.t:
        on = np.array(f.on, dtype=bool)
        for i0, i1 in runs(on):
            first = f.info[i0][0] if f.info[i0] else f.t[i0]
            out.append(RawEvent("road_obstacle", min(first, f.t[i0]), f.t[i1] + ctx.obstacle_every / 2, 0.6, (),
                                "static foreground"))
    return out


def rule_fire_smoke(ctx: SceneContext) -> list[RawEvent]:
    c = class_cfg(ctx.cfg, "fire_smoke")
    f = ctx.fire_flags
    if f is None or not f.t:
        return []
    on = fill_gaps(np.array(f.on, dtype=bool), 2)
    out = []
    for i0, i1 in runs(on):
        t0, t1 = f.t[i0] - ctx.fire_every / 2, f.t[i1] + ctx.fire_every / 2
        if t1 - t0 >= float(c.get("min_duration_sec", 2.0)):
            out.append(RawEvent("fire_smoke", t0, t1, 0.6, (), "classifier/heuristic"))
    return out


# ======================================================================
RULES = [
    ("stopped_vehicle", rule_stopped_vehicle, ("stopped_vehicle",)),
    ("wrong_way", rule_wrong_way, ("wrong_way",)),
    ("congestion", rule_congestion, ("congestion",)),
    ("jaywalking", rule_jaywalking, ("jaywalking",)),
    ("failure_to_yield", rule_failure_to_yield, ("failure_to_yield",)),
    ("signal", rule_signal_violations, ("red_light", "stop_line")),
    ("solid_line_crossing", rule_solid_line, ("solid_line_crossing",)),
    ("prohibited_turns", rule_prohibited_turns, ("illegal_turn", "illegal_u_turn")),
    ("u_turn", rule_u_turn, ("illegal_u_turn",)),
    ("collisions", rule_accident_near_miss, ("accident", "near_miss")),
    ("road_obstacle", rule_road_obstacle, ("road_obstacle",)),
    ("fire_smoke", rule_fire_smoke, ("fire_smoke",)),
]


def run_rules(ctx: SceneContext) -> list[RawEvent]:
    events: list[RawEvent] = []
    for name, fn, labels in RULES:
        if not any(class_enabled(ctx.cfg, lab) for lab in labels):
            continue
        try:
            found = fn(ctx)
            events.extend(e for e in found if class_enabled(ctx.cfg, e.label))
        except Exception as exc:  # noqa: BLE001 - one broken rule must not kill the others
            log.warning("rule %s failed: %s", name, exc, exc_info=log.isEnabledFor(logging.DEBUG))
    return events
