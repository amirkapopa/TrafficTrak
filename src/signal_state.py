"""Traffic-signal state from a configured signal-head ROI.

The ROI is split into lamp cells according to the layout (vertical: red at
the top).  For each cell we measure the brightness of its brightest pixels
and how well their hue matches the lamp colour; the lit lamp must dominate the
others by ``dominance_ratio``.  Ambiguous frames are ``unknown``.  States are
mode-filtered over ``smooth_window_sec``.  If too few samples of a video have
a known state, the red_light/stop_line rules are disabled for that video.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import SignalSpec

STATES = ("unknown", "red", "amber", "green")
_HUE = {"red": ((0, 12), (165, 180)), "amber": ((12, 35),), "green": ((40, 100),)}


def _hue_match(hsv: np.ndarray, colour: str) -> np.ndarray:
    h = hsv[..., 0]
    m = np.zeros(h.shape, dtype=bool)
    for lo, hi in _HUE.get(colour, ()):
        m |= (h >= lo) & (h <= hi)
    return m & (hsv[..., 1] >= 60)


def lamp_cells(roi_img: np.ndarray, layout: str, n: int) -> list[np.ndarray]:
    h, w = roi_img.shape[:2]
    cells = []
    for i in range(n):
        if layout == "horizontal":
            cells.append(roi_img[:, int(i * w / n):int((i + 1) * w / n)])
        else:
            cells.append(roi_img[int(i * h / n):int((i + 1) * h / n), :])
    return cells


def classify_signal(frame: np.ndarray, spec: SignalSpec, dominance: float = 1.35, min_bright: float = 120) -> str:
    x0, y0, x1, y1 = spec.roi
    roi = frame[max(y0, 0):y1, max(x0, 0):x1]
    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return "unknown"
    scores = {}
    for colour, cell in zip(spec.lamps, lamp_cells(roi, spec.layout, len(spec.lamps))):
        if cell.size == 0:
            continue
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2].reshape(-1).astype(np.float64)
        k = max(1, int(0.2 * v.size))
        top = np.partition(v, -k)[-k:]
        bright = float(top.mean())
        thr = np.partition(v, -k)[-k]
        mask = hsv[..., 2] >= thr
        hue_ok = float(_hue_match(hsv, colour)[mask].mean()) if mask.any() else 0.0
        scores[colour] = bright * (0.5 + 0.5 * hue_ok)
    if len(scores) < 2:
        return "unknown"
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 < min_bright or s1 < dominance * max(s2, 1e-6):
        return "unknown"
    return best if best in STATES else "unknown"


@dataclass
class SignalTimeline:
    signal_id: str
    t: list = field(default_factory=list)
    raw: list = field(default_factory=list)
    smooth_window: float = 1.0
    _states: np.ndarray | None = None

    def add(self, t: float, state: str) -> None:
        self.t.append(float(t))
        self.raw.append(state)
        self._states = None

    def finalize(self) -> np.ndarray:
        """Mode filter over a centred window (unknown only wins if nothing else)."""
        if self._states is not None:
            return self._states
        t = np.asarray(self.t)
        codes = np.array([STATES.index(s) for s in self.raw], dtype=np.int64)
        out = codes.copy()
        half = self.smooth_window / 2
        lo = np.searchsorted(t, t - half, side="left")
        hi = np.searchsorted(t, t + half, side="right")
        for i in range(len(t)):
            window = codes[lo[i]:hi[i]]
            counts = np.bincount(window, minlength=len(STATES))
            counts[0] = 0
            if counts.max() > 0 and counts.max() * 2 >= len(window) - np.sum(window == 0):
                out[i] = int(np.argmax(counts))
            else:
                out[i] = 0
        self._states = out
        return out

    @property
    def reliability(self) -> float:
        if not self.raw:
            return 0.0
        return float(np.mean(self.finalize() != 0))

    def state_at(self, t: float) -> str:
        if not self.t:
            return "unknown"
        states = self.finalize()
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        if i < 0:
            return "unknown"
        return STATES[int(states[i])]

    def is_state_throughout(self, state: str, t0: float, t1: float) -> bool:
        if not self.t:
            return False
        states = self.finalize()
        ts = np.asarray(self.t)
        code = STATES.index(state)
        i0 = max(int(np.searchsorted(ts, t0, side="right")) - 1, 0)
        i1 = int(np.searchsorted(ts, t1, side="right"))
        seg = states[i0:max(i1, i0 + 1)]
        return bool(len(seg) > 0 and np.all(seg == code))

    def next_change(self, t: float, to_state: str) -> float | None:
        """First time >= t at which the state becomes ``to_state``."""
        if not self.t:
            return None
        states = self.finalize()
        code = STATES.index(to_state)
        ts = np.asarray(self.t)
        i0 = int(np.searchsorted(ts, t, side="left"))
        hit = np.where(states[i0:] == code)[0]
        return float(ts[i0 + hit[0]]) if hit.size else None

    def intervals(self) -> list[tuple[float, float, str]]:
        states = self.finalize()
        out = []
        if not self.t:
            return out
        start = 0
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != states[start]:
                end_t = self.t[i] if i < len(states) else self.t[-1]
                out.append((self.t[start], end_t, STATES[int(states[start])]))
                start = i
        return out
