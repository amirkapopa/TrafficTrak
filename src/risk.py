"""Part B: causal accident-risk estimator.

``RiskEstimator.step`` sees one frame at a time and only uses the current and
earlier frames.  Per frame it runs the shared detector (every
``detect_every`` frames), updates an online ByteTracker, maintains short
kinematic histories and scores every nearby road-user pair:

* F1 conflict  - predicted closest approach within the horizon (TTC), scaled by
  closing speed and miss distance;
* F2 evasive   - abrupt deceleration / sudden heading change (either party);
* F3 violation - wrong-way motion, pedestrian on the carriageway, vehicle
  approaching a red signal at speed.

``p = sigmoid(bias + w1*F1 + w2*F2 + w3*F3 + w4*F1*F2)``.  With the default
weights a single cue stays below 0.5; only agreeing cues (a close, fast
conflict plus evasive action or a violation) push the score above 0.5.  The
frame hazard is smoothed with fast-attack / slow-decay dynamics, and values
>= ``arm_threshold`` are only released after the raw hazard has stayed high
for ``arm_hold_sec`` (hysteresis), so single-frame detector noise cannot raise
an alarm.  The weights are hand-set and documented, not fitted: no labelled
data was available.  ``scripts/eval_dev.py`` reports how they behave once the
team has annotated clips.
"""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from .config import class_cfg, geometry_config_path, load_config
from .detection import CLASS_GROUP, get_detector
from .determinism import set_determinism
from .geometry import Geometry, angle_diff, load_geometry
from .pipeline import load_scene_prior
from .signal_state import STATES, classify_signal
from .tracking import ByteTracker
from .video import frame_ok, valid_fps

log = logging.getLogger("traffictrak.risk")


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))


class _Kinematics:
    """Short causal history of one track's ground point."""

    def __init__(self, history_sec: float):
        self.history_sec = history_sec
        self.hist: deque = deque()      # (t, gx, gy, scale)
        self.speed: deque = deque()     # (t, speed_n)
        self.heading: deque = deque()   # (t, heading) while moving
        self.wrong_run = 0.0
        self.last_t = None

    def add(self, t: float, box: np.ndarray) -> None:
        gx, gy = (box[0] + box[2]) / 2, box[3]
        scale = max(math.sqrt(max(box[2] - box[0], 1) * max(box[3] - box[1], 1)), 4.0)
        self.hist.append((t, gx, gy, scale))
        while self.hist and t - self.hist[0][0] > self.history_sec:
            self.hist.popleft()
        vx, vy = self.velocity()
        sp = math.hypot(vx, vy) / scale
        self.speed.append((t, sp))
        while self.speed and t - self.speed[0][0] > self.history_sec:
            self.speed.popleft()
        if sp >= 0.4:
            self.heading.append((t, math.atan2(vy, vx)))
        while self.heading and t - self.heading[0][0] > self.history_sec:
            self.heading.popleft()
        self.last_t = t

    def velocity(self, window: float = 0.6) -> tuple[float, float]:
        if len(self.hist) < 3:
            return 0.0, 0.0
        t_now = self.hist[-1][0]
        pts = [h for h in self.hist if t_now - h[0] <= window]
        if len(pts) < 3:
            pts = list(self.hist)[-3:]
        arr = np.array(pts)
        tt = arr[:, 0] - arr[:, 0].mean()
        den = float((tt * tt).sum())
        if den <= 1e-9:
            return 0.0, 0.0
        return float((tt * (arr[:, 1] - arr[:, 1].mean())).sum() / den), float((tt * (arr[:, 2] - arr[:, 2].mean())).sum() / den)

    @property
    def span(self) -> float:
        return self.hist[-1][0] - self.hist[0][0] if len(self.hist) > 1 else 0.0

    @property
    def point(self) -> tuple[float, float]:
        return self.hist[-1][1], self.hist[-1][2]

    @property
    def scale(self) -> float:
        return self.hist[-1][3]

    def current_speed(self) -> float:
        return self.speed[-1][1] if self.speed else 0.0

    def decel(self, window: float = 1.0) -> tuple[float, float]:
        """(speed drop over the last ``window`` s, peak speed in that window)."""
        if not self.speed:
            return 0.0, 0.0
        t_now = self.speed[-1][0]
        recent = [s for (t, s) in self.speed if t_now - t <= window]
        peak = max(recent)
        return peak - self.speed[-1][1], peak

    def heading_change(self, window: float = 0.7) -> float:
        if len(self.heading) < 2:
            return 0.0
        t_now = self.heading[-1][0]
        recent = [h for (t, h) in self.heading if t_now - t <= window]
        if len(recent) < 2:
            return 0.0
        u = np.unwrap(np.array(recent))
        return float(u.max() - u.min())


class RiskEstimator:
    """Causal P(accident starts within the next 5 s) for each frame."""

    def __init__(self, config_path: str | None = None, geometry_path: str | None = None, detector=None,
                 overrides: dict | None = None):
        self.cfg = load_config(config_path, overrides)
        self.rc = self.cfg.get("risk", {})
        self.geometry_path = geometry_path or str(geometry_config_path())
        self._detector = detector
        self._detector_failed = False
        self.meta: dict = {}
        self.last_components: dict = {}
        self.reset({})

    # ------------------------------------------------------------------
    def _get_detector(self):
        if self._detector is None and not self._detector_failed:
            try:
                self._detector = get_detector(self.cfg)
            except Exception as exc:  # noqa: BLE001
                log.warning("risk estimator: detector unavailable (%s) - returning low constant risk", exc)
                self._detector_failed = True
        return self._detector

    def reset(self, meta: dict) -> None:
        """Clear all tracker and temporal state for a new video."""
        set_determinism(int(self.cfg.get("seed", 0)))
        self.meta = dict(meta or {})
        fps = self.meta.get("fps")
        self.fps = float(fps) if valid_fps(fps) else float(self.cfg.get("video", {}).get("default_fps", 25.0))
        self.width = int(self.meta.get("width") or 0)
        self.height = int(self.meta.get("height") or 0)
        self.geometry: Geometry | None = None
        if self.width > 0 and self.height > 0:
            self.geometry = load_geometry(self.geometry_path, self.width, self.height)
        self.tracker = ByteTracker(self.cfg, self.fps)
        self.kin: dict[int, _Kinematics] = {}
        self.frame_no = 0
        self.t_prev: float | None = None
        self.raw = 0.0
        self.value = 0.0
        self.armed = False
        self.high_since: float | None = None
        self.sig_hist: dict[str, deque] = {}
        self.last_components = {}
        self.prior_flow = load_scene_prior(self.cfg)[0]
        det = self._detector
        on_gpu = bool(getattr(det, "on_gpu", False)) if det is not None else None
        if on_gpu is None:
            from .detection import cuda_available

            on_gpu = cuda_available()
        self.detect_every = max(1, int(self.rc.get("detect_every_gpu", 1) if on_gpu else self.rc.get("detect_every_cpu", 3)))

    # ------------------------------------------------------------------
    def step(self, frame, t_sec: float) -> float:
        try:
            return self._step(frame, float(t_sec))
        except Exception as exc:  # noqa: BLE001 - never crash the harness
            log.warning("risk step failed at t=%.2f: %s", t_sec, exc)
            return float(min(max(self.value, 0.0), 1.0))

    def _step(self, frame, t: float) -> float:
        if not math.isfinite(t):
            return float(self.value)
        dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0)
        self.t_prev = t
        if not frame_ok(frame):
            return self._smooth(self.raw * 0.9, dt, t)
        if self.geometry is None:
            h, w = frame.shape[:2]
            self.width, self.height = w, h
            self.geometry = load_geometry(self.geometry_path, w, h)
        idx = self.frame_no
        self.frame_no += 1
        if idx % self.detect_every == 0:
            det = self._get_detector()
            if det is not None:
                dets = det(frame)
                self.tracker.update(dets, t, idx)
                active = self.tracker.active_tracks()
                alive = set()
                for tr in active:
                    if tr.last_t == t:
                        self.kin.setdefault(tr.track_id, _Kinematics(float(self.rc.get("history_sec", 2.0)))).add(t, tr.last_box)
                    alive.add(tr.track_id)
                for tid in [k for k in self.kin if k not in alive and t - (self.kin[k].last_t or t) > 2.0]:
                    del self.kin[tid]
                self._update_signals(frame, t)
                self.raw = self._hazard(active, t)
        return self._smooth(self.raw, dt, t)

    # ------------------------------------------------------------------
    def _smooth(self, h: float, dt: float, t: float) -> float:
        rc = self.rc
        tau = float(rc.get("tau_up_sec", 0.3)) if h > self.value else float(rc.get("tau_down_sec", 2.0))
        alpha = 1.0 - math.exp(-dt / tau) if dt > 0 else (1.0 if self.frame_no <= 1 else 0.0)
        self.value = self.value + alpha * (h - self.value)
        arm_thr = float(rc.get("arm_threshold", 0.5))
        if h >= arm_thr:
            self.high_since = t if self.high_since is None else self.high_since
            if t - self.high_since >= float(rc.get("arm_hold_sec", 0.3)):
                self.armed = True
        else:
            self.high_since = None
        if self.value < float(rc.get("disarm_threshold", 0.3)):
            self.armed = False
        out = self.value if self.armed else min(self.value, float(rc.get("unarmed_cap", 0.49)))
        return float(min(max(out, 0.0), 1.0))

    def _update_signals(self, frame, t: float) -> None:
        if not self.geometry or not self.geometry.signals:
            return
        sc = self.cfg.get("signal", {})
        for spec in self.geometry.signals:
            q = self.sig_hist.setdefault(spec.id, deque())
            q.append((t, classify_signal(frame, spec, float(sc.get("dominance_ratio", 1.35)), float(sc.get("min_lamp_brightness", 120)))))
            while q and t - q[0][0] > float(sc.get("smooth_window_sec", 1.0)):
                q.popleft()

    def _signal_state(self, sid) -> str:
        q = self.sig_hist.get(str(sid))
        if not q:
            return "unknown"
        counts = {s: 0 for s in STATES}
        for _, s in q:
            counts[s] += 1
        counts["unknown"] = 0
        best = max(STATES[1:], key=lambda s: counts[s])
        return best if counts[best] > 0 else "unknown"

    def _violation(self, tid: int, group: str, k: _Kinematics, dt_step: float) -> float:
        g = self.geometry
        p = k.point
        sp = k.current_speed()
        vx, vy = k.velocity()
        heading = math.atan2(vy, vx)
        score = 0.0
        if group in ("vehicle", "two_wheeler") and sp >= 1.0:
            ref = None
            if g is not None and g.lanes:
                lane = g.lane_of(p)
                ref = lane.heading_at(p) if lane else None
            elif self.prior_flow is not None:
                wc = class_cfg(self.cfg, "wrong_way")
                ang, conc, sup = self.prior_flow.dominant(p[0], p[1], self.width, self.height)
                if sup >= float(wc.get("auto_min_support", 6)) and conc >= float(wc.get("auto_min_concentration", 0.85)):
                    ref = ang
            if ref is not None and angle_diff(heading, ref) >= math.radians(135):
                k.wrong_run += dt_step
            else:
                k.wrong_run = 0.0
            if k.wrong_run >= 1.0:
                score = max(score, 1.0)
            if g is not None:
                for sl in g.stop_lines:
                    if self._signal_state(sl.signal) != "red":
                        continue
                    d = -sl.distance_past(p)
                    along = vx * sl.approach[0] + vy * sl.approach[1]
                    if along > 0 and 0 < d < 1.5 * along:
                        score = max(score, 0.8)
        if group == "person" and g is not None and g.carriageway:
            on = g.on_carriageway(np.array([p]))
            if on is not None and on[0] and not (g.crossings and g.in_any(g.crossings, np.array([p]))[0]):
                score = max(score, 0.7)
        return score

    def _evasive(self, k: _Kinematics) -> float:
        rc = self.rc
        drop, peak = k.decel(1.0)
        e = 0.0
        if peak >= 1.0:
            e = max(e, min(drop / float(rc.get("decel_ref", 2.5)), 1.0))
        if k.current_speed() >= 0.8:
            e = max(e, min(k.heading_change(0.7) / math.radians(float(rc.get("swerve_ref_deg", 40))), 1.0))
        return e

    def _hazard(self, active, t: float) -> float:
        rc = self.rc
        min_hist = float(rc.get("min_history_sec", 0.4))
        dt_step = self.detect_every / self.fps
        items = []
        for tr in active:
            k = self.kin.get(tr.track_id)
            group = CLASS_GROUP.get(tr.cls_name, "other")
            if k is None or k.span < min_hist or group not in ("vehicle", "two_wheeler", "person"):
                continue
            items.append((tr.track_id, group, k, self._evasive(k), self._violation(tr.track_id, group, k, dt_step)))
        bias = float(rc.get("bias", -4.0))
        w1, w2 = float(rc.get("w_conflict", 3.0)), float(rc.get("w_evasive", 2.0))
        w3, w4 = float(rc.get("w_violation", 1.5)), float(rc.get("w_interaction", 1.5))
        best = _sigmoid(bias)
        comp = {"conflict": 0.0, "evasive": 0.0, "violation": 0.0, "pair": None}
        for _tid, _g, _k, ev, vio in items:
            p = _sigmoid(bias + 0.5 * w2 * ev + w3 * vio * 0.5)
            if p > best:
                best = p
                comp = {"conflict": 0.0, "evasive": ev, "violation": vio, "pair": None}
        horizon = float(rc.get("horizon_sec", 4.0))
        R = float(rc.get("conflict_radius", 0.8))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ta, ga, ka, ea, va = items[i]
                tb, gb, kb, eb, vb = items[j]
                if ga == "person" and gb == "person":
                    continue
                s = 0.5 * (ka.scale + kb.scale)
                pa, pb = np.array(ka.point), np.array(kb.point)
                rel = (pb - pa) / s
                dist = float(np.linalg.norm(rel))
                if dist > float(rc.get("max_pair_distance", 25.0)):
                    continue  # coarse cutoff; the TTC horizon does the real gating
                vel = (np.array(kb.velocity()) - np.array(ka.velocity())) / s
                vv = float(vel @ vel)
                if vv < 1e-6 or dist < 1e-6:
                    continue
                closing = -float(rel @ vel) / dist
                if closing <= 0:
                    continue
                t_star = min(max(-float(rel @ vel) / vv, 0.0), horizon)
                dmin = float(np.linalg.norm(rel + vel * t_star))
                if dmin >= R:
                    continue
                f1 = max(0.0, 1.0 - t_star / float(rc.get("ttc_scale_sec", 3.0))) \
                    * min(closing / float(rc.get("closing_ref", 2.0)), 1.0) * math.sqrt(max(0.0, 1.0 - dmin / R))
                f2 = max(ea, eb)
                f3 = max(va, vb)
                p = _sigmoid(bias + w1 * f1 + w2 * f2 + w3 * f3 + w4 * f1 * f2)
                if p > best:
                    best = p
                    comp = {"conflict": f1, "evasive": f2, "violation": f3, "pair": (ta, tb)}
        self.last_components = comp
        return best


def risk_curve(video_path: str, estimator: RiskEstimator | None = None, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Offline helper for visualisation: step through a video causally."""
    import cv2

    from .video import probe

    est = estimator or RiskEstimator()
    meta = probe(video_path)
    if not meta.ok:
        return np.zeros(0), np.zeros(0)
    est.reset(meta.as_dict())
    cap = cv2.VideoCapture(video_path)
    ts, rs = [], []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / meta.fps
            rs.append(est.step(frame, t))
            ts.append(t)
            idx += 1
            if progress and meta.n_frames and idx % 25 == 0:
                progress(min(idx / meta.n_frames, 1.0), "risk")
    finally:
        cap.release()
    return np.array(ts), np.array(rs)
