"""Open-weight object detection: YOLOX (Apache-2.0) exported to ONNX.

Pre/post-processing reimplements the reference ONNX demo of
Megvii-BaseDetection/YOLOX (Apache-2.0): letterbox to 640x640 with pad value
114, raw BGR float input, grid decoding of the (1, 8400, 85) head.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import resolve_path, weights_dir

log = logging.getLogger("traffictrak.detection")

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
    "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket", "bottle",
    "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
    "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy_bear", "hair_drier", "toothbrush",
]

# Classes relevant to road traffic and the group each belongs to.  Association
# in the tracker is only allowed within a group (car <-> truck confusions are
# common, car <-> person never).
CLASS_GROUP = {
    "person": "person",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "traffic_light": "signal",
}
KEEP_IDS = np.array(sorted(COCO_NAMES.index(n) for n in CLASS_GROUP), dtype=np.int64)
GROUP_IDS = {g: i for i, g in enumerate(sorted(set(CLASS_GROUP.values())))}
VEHICLE_CLASSES = ("car", "bus", "truck", "motorcycle")
ROAD_USER_GROUPS = ("vehicle", "two_wheeler", "person")


@dataclass
class Detections:
    boxes: np.ndarray    # (N, 4) float32 xyxy in pixels
    scores: np.ndarray   # (N,) float32
    classes: np.ndarray  # (N,) int64 COCO ids

    @staticmethod
    def empty() -> Detections:
        return Detections(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def names(self) -> list[str]:
        return [COCO_NAMES[int(c)] for c in self.classes]


def class_name(coco_id: int) -> str:
    return COCO_NAMES[int(coco_id)]


def group_of(name: str) -> str:
    return CLASS_GROUP.get(name, "other")


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, pad bottom/right with 114 (YOLOX convention)."""
    h, w = img.shape[:2]
    th, tw = size
    r = min(th / h, tw / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    padded = np.full((th, tw, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return padded, r


def _make_grids(h: int, w: int, strides=(8, 16, 32)) -> tuple[np.ndarray, np.ndarray]:
    grids, exp_strides = [], []
    for s in strides:
        hs, ws = h // s, w // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        exp_strides.append(np.full((grid.shape[0], 1), s))
    return np.concatenate(grids, 0).astype(np.float32), np.concatenate(exp_strides, 0).astype(np.float32)


def decode_yolox(raw: np.ndarray, input_hw: tuple[int, int], grids=None) -> np.ndarray:
    """Decode a raw (N, 85) YOLOX head output into cx, cy, w, h, obj, cls..."""
    out = raw.astype(np.float32).copy()
    if grids is None:
        grids = _make_grids(*input_hw)
    grid, strides = grids
    out[:, :2] = (out[:, :2] + grid) * strides
    out[:, 2:4] = np.exp(np.clip(out[:, 2:4], -20, 20)) * strides
    return out


def postprocess(decoded: np.ndarray, ratio: float, frame_hw: tuple[int, int], score_thr: float,
                nms_iou: float, min_area: float) -> Detections:
    obj = decoded[:, 4]
    cls_scores = decoded[:, 5:][:, KEEP_IDS]
    best = np.argmax(cls_scores, axis=1)
    scores = obj * cls_scores[np.arange(len(best)), best]
    keep = scores >= score_thr
    if not np.any(keep):
        return Detections.empty()
    d = decoded[keep]
    scores = scores[keep].astype(np.float32)
    classes = KEEP_IDS[best[keep]]
    cx, cy, w, h = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / ratio
    fh, fw = frame_hw
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, fw - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, fh - 1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valid = area >= min_area
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    if len(scores) == 0:
        return Detections.empty()
    # Group-wise NMS (car/bus/truck duplicates collapse to one box).
    groups = np.array([GROUP_IDS[group_of(COCO_NAMES[c])] for c in classes], dtype=np.float32)
    offset = groups[:, None] * (max(fw, fh) + 1.0) * 2.0
    shifted = boxes + offset
    xywh = np.stack([shifted[:, 0], shifted[:, 1], shifted[:, 2] - shifted[:, 0], shifted[:, 3] - shifted[:, 1]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), float(score_thr), float(nms_iou))
    idx = np.array(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return Detections.empty()
    # deterministic order: score desc, then x1, then y1
    order = sorted(idx.tolist(), key=lambda i: (-float(scores[i]), float(boxes[i, 0]), float(boxes[i, 1])))
    order = np.array(order, dtype=np.int64)
    return Detections(boxes[order].astype(np.float32), scores[order], classes[order].astype(np.int64))


class YoloxOnnxDetector:
    """YOLOX ONNX detector with deterministic onnxruntime settings."""

    def __init__(self, model_path: Path, prefer_gpu: bool = True, intra_op_threads: int = 0,
                 score_threshold: float = 0.1, nms_iou: float = 0.55, min_box_area_frac: float = 2e-5):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        so.use_deterministic_compute = True
        if intra_op_threads:
            so.intra_op_num_threads = int(intra_op_threads)
        providers: list = ["CPUExecutionProvider"]
        if prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = [("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "DEFAULT"}),
                         "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)
        self.on_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        h = shape[2] if isinstance(shape[2], int) else 640
        w = shape[3] if isinstance(shape[3], int) else 640
        self.input_hw = (h, w)
        self.grids = _make_grids(h, w)
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.min_box_area_frac = min_box_area_frac
        self._lock = threading.Lock()
        log.info("detector %s loaded (gpu=%s)", self.model_path.name, self.on_gpu)

    def __call__(self, frame: np.ndarray) -> Detections:
        if frame is None or frame.size == 0:
            return Detections.empty()
        img, ratio = letterbox(frame, self.input_hw)
        blob = np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32))
        with self._lock:
            raw = self.session.run(None, {self.input_name: blob})[0][0]
        decoded = decode_yolox(raw, self.input_hw, self.grids)
        fh, fw = frame.shape[:2]
        return postprocess(decoded, ratio, (fh, fw), self.score_threshold, self.nms_iou,
                           self.min_box_area_frac * fh * fw)


def cuda_available() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def resolve_model_path(model: str, prefer_gpu: bool) -> Path | None:
    """Pick the detector weights.  ``auto``: YOLOX-M on GPU, YOLOX-S on CPU."""
    wdir = weights_dir()
    if model not in ("auto", "yolox_m", "yolox_s"):
        p = resolve_path(model)
        return p if p is not None and p.is_file() else None
    if model == "auto":
        order = ["yolox_m", "yolox_s"] if (prefer_gpu and cuda_available()) else ["yolox_s", "yolox_m"]
    else:
        order = [model, "yolox_s" if model == "yolox_m" else "yolox_m"]
    for name in order:
        p = wdir / f"{name}.onnx"
        if p.is_file():
            return p
    return None


_DETECTORS: dict[tuple, YoloxOnnxDetector] = {}
_DET_LOCK = threading.Lock()


def get_detector(cfg: dict) -> YoloxOnnxDetector:
    """Return a cached detector (weights are loaded once per process)."""
    dcfg = cfg.get("detector", {})
    prefer_gpu = bool(dcfg.get("prefer_gpu", True))
    path = resolve_model_path(str(dcfg.get("model", "auto")), prefer_gpu)
    if path is None:
        raise FileNotFoundError(
            f"no detector weights found in {weights_dir()} - run `bash weights/download.sh` before evaluation")
    key = (str(path), prefer_gpu, float(dcfg.get("score_threshold", 0.1)), float(dcfg.get("nms_iou", 0.55)))
    with _DET_LOCK:
        if key not in _DETECTORS:
            _DETECTORS[key] = YoloxOnnxDetector(
                path, prefer_gpu=prefer_gpu, intra_op_threads=int(dcfg.get("intra_op_threads", 0) or 0),
                score_threshold=float(dcfg.get("score_threshold", 0.1)), nms_iou=float(dcfg.get("nms_iou", 0.55)),
                min_box_area_frac=float(dcfg.get("min_box_area_frac", 2e-5)))
        return _DETECTORS[key]
