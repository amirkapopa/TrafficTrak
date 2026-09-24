"""Frame-level scene monitors that run inside the main decoding pass.

* ``ObstacleMonitor`` - static foreground (debris / fallen load) on the
  carriageway that is not explained by any detected road user.
* ``FireSmokeMonitor`` - optional open-weight ONNX image classifier
  (``weights/fire_smoke.onnx``) and/or a strict colour+flicker heuristic.

Both work on small downscaled frames and are sampled sparsely, so their cost
is negligible compared with detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label as cc_label

from .config import resolve_path
from .geometry import Geometry

log = logging.getLogger("traffictrak.monitors")


@dataclass
class TimedFlag:
    t: list = field(default_factory=list)
    on: list = field(default_factory=list)
    info: list = field(default_factory=list)

    def add(self, t: float, on: bool, info=None) -> None:
        self.t.append(float(t))
        self.on.append(bool(on))
        self.info.append(info)


def _block_view(img: np.ndarray, b: int) -> np.ndarray:
    h, w = img.shape[:2]
    hb, wb = h // b, w // b
    return img[:hb * b, :wb * b].reshape(hb, b, wb, b).swapaxes(1, 2)


def block_ncc(a: np.ndarray, b: np.ndarray, blk: int) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise normalised cross-correlation and absolute mean difference."""
    va = _block_view(a.astype(np.float32), blk).reshape(a.shape[0] // blk, a.shape[1] // blk, -1)
    vb = _block_view(b.astype(np.float32), blk).reshape(va.shape)
    ma, mb = va.mean(-1, keepdims=True), vb.mean(-1, keepdims=True)
    da, db = va - ma, vb - mb
    num = (da * db).sum(-1)
    den = np.sqrt((da * da).sum(-1) * (db * db).sum(-1))
    flat = den < 1e-3 * blk * blk * 25  # low-texture blocks: NCC meaningless
    ncc = np.where(flat, 1.0, num / np.maximum(den, 1e-6))
    return ncc, np.abs(ma - mb)[..., 0]


class ObstacleMonitor:
    """Detect persistent, static, unexplained foreground on the road."""

    def __init__(self, cfg: dict, geometry: Geometry, width: int, height: int,
                 background: np.ndarray | None = None):
        c = cfg["classes"]["road_obstacle"]
        self.enabled = bool(c.get("enabled", True) and c.get("static_objects", True))
        self.every = float(c.get("sample_every_sec", 0.5))
        self.work_w = int(c.get("work_width", 320))
        self.blk = int(c.get("block", 8))
        self.ncc_changed = float(c.get("ncc_changed", 0.45))
        self.abs_changed = float(c.get("abs_changed", 28))
        self.static_ncc = float(c.get("static_ncc", 0.8))
        self.min_static = float(c.get("min_static_sec", 3.0))
        self.min_blocks = int(c.get("min_blocks", 2))
        self.max_area = float(c.get("max_area_frac", 0.05))
        self.learn_sec = float(c.get("learn_background_sec", 12.0))
        self.scale = self.work_w / float(width)
        self.work_h = int(round(height * self.scale))
        self.width, self.height = width, height
        self.flags = TimedFlag()
        self.next_t = 0.0
        self.learn: list[np.ndarray] = []
        self.bg: np.ndarray | None = None
        self.prev: np.ndarray | None = None
        self.since: np.ndarray | None = None
        regions = geometry.obstacle_regions or geometry.carriageway
        if not regions:
            self.enabled = False  # precision: never search debris without a calibrated road mask
        self.block_mask = None
        if self.enabled:
            hb, wb = self.work_h // self.blk, self.work_w // self.blk
            ys, xs = np.mgrid[0:hb, 0:wb]
            centres = np.stack([(xs.ravel() + 0.5) * self.blk / self.scale, (ys.ravel() + 0.5) * self.blk / self.scale], 1)
            inside = geometry.in_any(regions, centres)
            excl = geometry.in_any(geometry.exclusion_zones + geometry.sidewalks, centres)
            self.block_mask = (inside & ~excl).reshape(hb, wb)
            if background is not None:
                bg = cv2.resize(background, (self.work_w, self.work_h), interpolation=cv2.INTER_AREA)
                self.bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) if bg.ndim == 3 else bg

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (self.work_w, self.work_h), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    def update(self, frame: np.ndarray, t: float, boxes: np.ndarray) -> None:
        if not self.enabled or t + 1e-9 < self.next_t:
            return
        self.next_t = t + self.every
        g = self._prep(frame)
        if self.bg is None:
            self.learn.append(g)
            if t >= self.learn_sec and len(self.learn) >= 5:
                self.bg = np.median(np.stack(self.learn), axis=0).astype(np.uint8)
                self.learn = []
            self.prev = g
            return
        hb, wb = self.block_mask.shape
        ncc_bg, _ = block_ncc(g, self.bg, self.blk)
        gain = float(np.median(g)) / max(float(np.median(self.bg)), 1.0)
        diff_bg = np.abs(_block_view(g.astype(np.float32), self.blk).mean((-1, -2))
                         - gain * _block_view(self.bg.astype(np.float32), self.blk).mean((-1, -2)))
        changed = (ncc_bg < self.ncc_changed) | (diff_bg > self.abs_changed)
        ncc_prev, _ = block_ncc(g, self.prev if self.prev is not None else g, self.blk)
        static = ncc_prev >= self.static_ncc
        covered = np.zeros((hb, wb), dtype=bool)
        for b in np.asarray(boxes).reshape(-1, 4):
            x0, y0, x1, y1 = b * self.scale
            pad_x, pad_y = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
            bx0, by0 = int(max((x0 - pad_x) // self.blk, 0)), int(max((y0 - pad_y) // self.blk, 0))
            bx1, by1 = int(min((x1 + pad_x) // self.blk + 1, wb)), int(min((y1 + pad_y) // self.blk + 1, hb))
            covered[by0:by1, bx0:bx1] = True
        cand = changed & static & ~covered & self.block_mask
        if self.since is None:
            self.since = np.full((hb, wb), np.nan)
        self.since = np.where(cand, np.where(np.isnan(self.since), t, self.since), np.nan)
        persistent = cand & ((t - np.nan_to_num(self.since, nan=t)) >= self.min_static)
        on, info = False, None
        if persistent.any():
            lab, n = cc_label(persistent)
            max_blocks = self.max_area * hb * wb
            for i in range(1, n + 1):
                size = int((lab == i).sum())
                if self.min_blocks <= size <= max_blocks:
                    ys, xs = np.where(lab == i)
                    first = float(np.nanmin(self.since[lab == i]))
                    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64) * self.blk / self.scale
                    if not on or first < info[0]:
                        info = (first, box.tolist())
                    on = True
        # slow background adaptation on unchanged, uncovered blocks
        adapt = ~changed & ~covered
        if adapt.any():
            m = np.kron(adapt, np.ones((self.blk, self.blk), dtype=bool))
            full = np.zeros(g.shape, dtype=bool)
            full[:m.shape[0], :m.shape[1]] = m
            self.bg = np.where(full, (0.95 * self.bg + 0.05 * g).astype(np.uint8), self.bg)
        self.prev = g
        self.flags.add(t, on, info)


class FireSmokeMonitor:
    """Optional fire/smoke evidence (classifier and/or strict heuristic)."""

    def __init__(self, cfg: dict):
        c = cfg["classes"]["fire_smoke"]
        self.cfg = c
        self.every = float(c.get("sample_every_sec", 0.5))
        self.flags = TimedFlag()
        self.next_t = 0.0
        self.session = None
        mode = str(c.get("mode", "auto")) if c.get("enabled", True) else "off"
        path = resolve_path(c.get("classifier_path"))
        have_model = path is not None and Path(path).is_file()
        if mode in ("auto", "classifier") and have_model:
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.log_severity_level = 3
                so.use_deterministic_compute = True
                providers = ["CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                self.session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
            except Exception as exc:  # noqa: BLE001
                log.warning("fire/smoke classifier unusable: %s", exc)
                self.session = None
        self.use_classifier = self.session is not None
        self.use_heuristic = mode == "heuristic"
        if mode == "classifier" and not self.use_classifier:
            log.warning("fire_smoke mode=classifier but %s is missing - class disabled", path)
        self.enabled = self.use_classifier or self.use_heuristic
        self.prev_v: np.ndarray | None = None

    def _classify(self, frame: np.ndarray) -> float:
        c = self.cfg
        w, h = c.get("classifier_input", [224, 224])
        img = cv2.resize(frame, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        if c.get("classifier_rgb", True):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array(c.get("classifier_mean", [0, 0, 0]), np.float32)) / np.array(c.get("classifier_std", [1, 1, 1]), np.float32)
        out = self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None]})[0].reshape(-1)
        out = out.astype(np.float64)
        if out.min() < 0 or out.sum() > 1.0001:
            e = np.exp(out - out.max())
            out = e / e.sum()
        return float(sum(out[i] for i in c.get("classifier_positive_indices", [0]) if i < len(out)))

    def _heuristic(self, frame: np.ndarray, moving_boxes: np.ndarray) -> bool:
        small = cv2.resize(frame, (320, int(round(frame.shape[0] * 320 / frame.shape[1]))), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        b, g, r = small[..., 0].astype(np.int16), small[..., 1].astype(np.int16), small[..., 2].astype(np.int16)
        fire = (hsv[..., 0] <= 35) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 200) & (r > g) & (g > b)
        s = 320 / frame.shape[1]
        for bx in np.asarray(moving_boxes).reshape(-1, 4):
            x0, y0, x1, y1 = (bx * s).astype(int)
            fire[max(y0, 0):y1, max(x0, 0):x1] = False
        v = hsv[..., 2].astype(np.float32)
        flicker = 0.0
        if self.prev_v is not None and fire.any():
            flicker = float(np.mean(np.abs(v - self.prev_v)[fire] > 30))
        self.prev_v = v
        area = float(fire.mean())
        return area >= float(self.cfg.get("heuristic_min_area_frac", 0.0015)) and \
            flicker >= float(self.cfg.get("heuristic_flicker_frac", 0.25))

    def update(self, frame: np.ndarray, t: float, moving_boxes: np.ndarray) -> None:
        if not self.enabled or t + 1e-9 < self.next_t:
            return
        self.next_t = t + self.every
        on = True
        if self.use_classifier:
            on = self._classify(frame) >= float(self.cfg.get("classifier_threshold", 0.8))
        if self.use_heuristic:
            on = on and self._heuristic(frame, moving_boxes)
        self.flags.add(t, on)
