"""Multi-object tracking.

``ByteTracker`` is a clean-room reimplementation of the association scheme of
ByteTrack (Zhang et al., ECCV 2022; reference code MIT-licensed,
https://github.com/ifzhang/ByteTrack): high-score boxes are matched first,
low-score boxes then rescue tracks that would otherwise be lost.  The Kalman
filter is a textbook constant-velocity filter written for variable time steps
(frame stride may change).  Association is restricted to compatible class
groups.  Everything is deterministic (Hungarian assignment, stable ordering).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import CLASS_GROUP, COCO_NAMES, Detections


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


class KalmanBox:
    """Constant-velocity Kalman filter on [cx, cy, w, h] with variable dt."""

    STD_POS = 1.0 / 20.0
    STD_VEL = 1.0 / 160.0

    def __init__(self, box: np.ndarray):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        s = np.array([2 * self.STD_POS * h, 2 * self.STD_POS * h, 2 * self.STD_POS * h, 2 * self.STD_POS * h,
                      10 * self.STD_VEL * h, 10 * self.STD_VEL * h, 10 * self.STD_VEL * h, 10 * self.STD_VEL * h])
        self.P = np.diag(s ** 2)

    def predict(self, dt_frames: float) -> None:
        dt = max(dt_frames, 0.0)
        if dt == 0:
            return
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        h = max(self.x[3], 1.0)
        q_pos = (self.STD_POS * h) ** 2 * dt
        q_vel = (self.STD_VEL * h) ** 2 * dt
        Q = np.diag([q_pos] * 4 + [q_vel] * 4)
        self.x = F @ self.x
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self.P = F @ self.P @ F.T + Q

    def update(self, box: np.ndarray) -> None:
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                      max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)])
        H = np.zeros((4, 8))
        H[:4, :4] = np.eye(4)
        h = max(self.x[3], 1.0)
        R = np.diag([(self.STD_POS * h) ** 2] * 4)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    def velocity(self) -> np.ndarray:
        """Velocity of the box centre in pixels per frame."""
        return self.x[4:6].copy()


TRACKED, LOST, REMOVED = 0, 1, 2


@dataclass
class Track:
    track_id: int
    kf: KalmanBox
    start_t: float
    last_t: float
    last_frame: int
    pred_t: float = 0.0          # time the Kalman state currently refers to
    state: int = TRACKED
    activated: bool = False
    hits: int = 1
    cls_votes: dict = field(default_factory=dict)
    # history of real observations
    t_hist: list = field(default_factory=list)
    f_hist: list = field(default_factory=list)
    box_hist: list = field(default_factory=list)
    score_hist: list = field(default_factory=list)
    cls_hist: list = field(default_factory=list)

    @property
    def cls_name(self) -> str:
        if not self.cls_votes:
            return "unknown"
        return min(self.cls_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    @property
    def group(self) -> str:
        return CLASS_GROUP.get(self.cls_name, "other")

    def record(self, t: float, frame_idx: int, box: np.ndarray, score: float, cls_name: str) -> None:
        self.t_hist.append(t)
        self.f_hist.append(frame_idx)
        self.box_hist.append(np.asarray(box, dtype=np.float32))
        self.score_hist.append(float(score))
        self.cls_hist.append(cls_name)
        self.cls_votes[cls_name] = self.cls_votes.get(cls_name, 0.0) + float(score)

    @property
    def last_box(self) -> np.ndarray:
        return self.box_hist[-1]


class ByteTracker:
    """ByteTrack-style tracker working in wall-clock time (supports any stride)."""

    def __init__(self, cfg: dict, fps: float):
        tc = cfg.get("tracker", {}) if "tracker" in cfg else cfg
        self.high = float(tc.get("high_thresh", 0.45))
        self.low = float(tc.get("low_thresh", 0.10))
        self.new_thresh = float(tc.get("new_track_thresh", 0.55))
        self.match_iou = float(tc.get("match_iou", 0.2))
        self.second_iou = float(tc.get("second_match_iou", 0.45))
        self.unconf_iou = float(tc.get("unconfirmed_match_iou", 0.3))
        self.max_lost = float(tc.get("max_lost_sec", 1.5))
        self.fps = fps if fps and fps > 0 else 25.0
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self.next_id = 1
        self.n_updates = 0
        self.last_t: float | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _compatible(track_group: str, det_names: list[str]) -> np.ndarray:
        return np.array([CLASS_GROUP.get(n, "other") == track_group for n in det_names], dtype=bool)

    def _associate(self, tracks: list[Track], boxes: np.ndarray, names: list[str], min_iou: float):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        tboxes = np.stack([t.kf.box() for t in tracks])
        ious = iou_matrix(tboxes, boxes)
        for i, t in enumerate(tracks):
            ious[i, ~self._compatible(t.group, names)] = 0.0
        cost = 1.0 - ious
        cost[ious < min_iou] = 1e5
        rows, cols = linear_sum_assignment(cost)
        matches, used_t, used_d = [], set(), set()
        for r, c in zip(rows.tolist(), cols.tolist()):
            if cost[r, c] >= 1e4:
                continue
            matches.append((r, c))
            used_t.add(r)
            used_d.add(c)
        un_t = [i for i in range(len(tracks)) if i not in used_t]
        un_d = [j for j in range(len(boxes)) if j not in used_d]
        return matches, un_t, un_d

    def _apply(self, track: Track, box, score, name, t, frame_idx) -> None:
        track.kf.update(box)
        track.last_t = t
        track.last_frame = frame_idx
        track.hits += 1
        track.state = TRACKED
        track.record(t, frame_idx, box, score, name)

    def update(self, dets: Detections, t: float, frame_idx: int) -> list[Track]:
        """Update with detections of one processed frame; returns tracks updated now."""
        self.n_updates += 1
        names_all = [COCO_NAMES[int(c)] for c in dets.classes]
        # traffic lights are not tracked
        use = np.array([CLASS_GROUP.get(n, "other") not in ("signal", "other") for n in names_all], dtype=bool)
        boxes, scores = dets.boxes[use], dets.scores[use]
        names = [n for n, u in zip(names_all, use) if u]

        for tr in self.tracks:
            tr.kf.predict((t - tr.pred_t) * self.fps)
            tr.pred_t = t

        high = scores >= self.high
        low = (scores >= self.low) & ~high
        hi_idx = np.where(high)[0]
        lo_idx = np.where(low)[0]

        confirmed = [tr for tr in self.tracks if tr.activated]
        unconfirmed = [tr for tr in self.tracks if not tr.activated]
        updated: list[Track] = []

        # 1) confirmed (tracked + lost) vs high-score boxes
        m1, un_t1, un_d1 = self._associate(confirmed, boxes[hi_idx], [names[i] for i in hi_idx], self.match_iou)
        for r, c in m1:
            j = hi_idx[c]
            self._apply(confirmed[r], boxes[j], scores[j], names[j], t, frame_idx)
            updated.append(confirmed[r])
        # 2) remaining *tracked* tracks vs low-score boxes
        remain = [confirmed[i] for i in un_t1 if confirmed[i].state == TRACKED]
        m2, un_t2, _ = self._associate(remain, boxes[lo_idx], [names[i] for i in lo_idx], self.second_iou)
        for r, c in m2:
            j = lo_idx[c]
            self._apply(remain[r], boxes[j], scores[j], names[j], t, frame_idx)
            updated.append(remain[r])
        for i in un_t2:
            remain[i].state = LOST
        # 3) unconfirmed vs remaining high boxes
        left_hi = [hi_idx[c] for c in un_d1]
        m3, un_t3, un_d3 = self._associate(unconfirmed, boxes[left_hi] if left_hi else np.zeros((0, 4)),
                                           [names[i] for i in left_hi], self.unconf_iou)
        for r, c in m3:
            j = left_hi[c]
            unconfirmed[r].activated = True
            self._apply(unconfirmed[r], boxes[j], scores[j], names[j], t, frame_idx)
            updated.append(unconfirmed[r])
        for i in un_t3:
            unconfirmed[i].state = REMOVED
        # 4) new tracks
        for c in un_d3:
            j = left_hi[c]
            if scores[j] < self.new_thresh:
                continue
            tr = Track(self.next_id, KalmanBox(boxes[j]), start_t=t, last_t=t, last_frame=frame_idx,
                       pred_t=t, activated=(self.n_updates == 1))
            tr.record(t, frame_idx, boxes[j], scores[j], names[j])
            self.next_id += 1
            self.tracks.append(tr)
            if tr.activated:
                updated.append(tr)
        # 5) retire
        alive = []
        for tr in self.tracks:
            if tr.state == LOST and t - tr.last_t > self.max_lost:
                tr.state = REMOVED
            if tr.state == REMOVED:
                if tr.activated:
                    self.finished.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        self.last_t = t
        return sorted(updated, key=lambda tr: tr.track_id)

    def active_tracks(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.activated and tr.state == TRACKED]

    def all_tracks(self) -> list[Track]:
        """Every confirmed track seen so far (finished + alive), sorted by id."""
        out = list(self.finished) + [tr for tr in self.tracks if tr.activated]
        return sorted(out, key=lambda tr: tr.track_id)


def track_duration(tr: Track) -> float:
    return tr.t_hist[-1] - tr.t_hist[0] if tr.t_hist else 0.0


def _still(tr: Track, head: bool, window_sec: float = 1.0, max_move: float = 0.3) -> bool:
    """True if the box centre moves < max_move box-scales within the first
    (head) or last (tail) ``window_sec`` of the track."""
    t = tr.t_hist
    idx = [i for i in range(len(t)) if (t[i] - t[0] if head else t[-1] - t[i]) <= window_sec]
    if len(idx) < 2:
        return True
    boxes = np.stack([tr.box_hist[i] for i in idx])
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    scale = float(np.sqrt(max((boxes[:, 2] - boxes[:, 0]).mean(), 1.0) * max((boxes[:, 3] - boxes[:, 1]).mean(), 1.0)))
    return float(np.linalg.norm(c.max(0) - c.min(0))) < max_move * scale


def stitch_stationary(tracks: list[Track], max_gap: float, min_iou: float) -> list[Track]:
    """Merge fragments of the same (almost) stationary object.

    A stationary vehicle occluded by passing traffic often loses its track.  If
    a later track of the same group starts where an earlier one ended (box IoU
    >= ``min_iou``) within ``max_gap`` seconds, the two are joined.
    """
    tracks = sorted(tracks, key=lambda tr: (tr.t_hist[0], tr.track_id))
    out: list[Track] = []
    for tr in tracks:
        best, best_iou = None, min_iou
        if not _still(tr, head=True):
            out.append(tr)
            continue
        for prev in out:
            if prev.group != tr.group or not _still(prev, head=False):
                continue
            gap = tr.t_hist[0] - prev.t_hist[-1]
            if gap <= 0 or gap > max_gap:
                continue
            iou = float(iou_matrix(prev.last_box[None], tr.box_hist[0][None])[0, 0])
            if iou >= best_iou:
                best, best_iou = prev, iou
        if best is None:
            out.append(tr)
            continue
        best.t_hist += tr.t_hist
        best.f_hist += tr.f_hist
        best.box_hist += tr.box_hist
        best.score_hist += tr.score_hist
        best.cls_hist += tr.cls_hist
        for k, v in tr.cls_votes.items():
            best.cls_votes[k] = best.cls_votes.get(k, 0.0) + v
        best.last_t = tr.last_t
    return sorted(out, key=lambda tr: tr.track_id)
