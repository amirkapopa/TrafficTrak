"""Video metadata, robust sequential frame reading and timestamps.

Timestamps are always ``frame_index / fps`` relative to the first frame, which
is deterministic and matches how evaluators convert frame indices to seconds.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("traffictrak.video")


@dataclass
class VideoMeta:
    path: str
    fps: float
    n_frames: int          # container frame count (0 if unknown)
    width: int
    height: int
    fps_reliable: bool
    ok: bool
    error: str = ""

    @property
    def duration(self) -> float:
        if self.fps > 0 and self.n_frames > 0:
            return self.n_frames / self.fps
        return 0.0

    def as_dict(self) -> dict:
        return {
            "video_id": os.path.splitext(os.path.basename(self.path))[0],
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "n_frames": self.n_frames,
        }


def valid_fps(fps: float | None) -> bool:
    return fps is not None and math.isfinite(fps) and 1.0 <= fps <= 240.0


def _estimate_fps(cap: cv2.VideoCapture, default_fps: float) -> float:
    """Estimate fps from decoder timestamps when the container reports none."""
    stamps = []
    for _ in range(30):
        if not cap.grab():
            break
        stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    if len(stamps) >= 3 and stamps[-1] > stamps[0]:
        fps = (len(stamps) - 1) * 1000.0 / (stamps[-1] - stamps[0])
        if valid_fps(fps):
            return fps
    return default_fps


def probe(path: str, default_fps: float = 25.0) -> VideoMeta:
    """Read container metadata without decoding the whole file."""
    if not path or not os.path.isfile(path):
        return VideoMeta(str(path), default_fps, 0, 0, 0, False, False, "file not found")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return VideoMeta(path, default_fps, 0, 0, 0, False, False, "cannot open video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        reliable = valid_fps(fps)
        if not reliable:
            fps = _estimate_fps(cap, default_fps)
            log.warning("%s: unreliable fps metadata, using %.3f", path, fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                return VideoMeta(path, fps, 0, 0, 0, reliable, False, "no decodable frames")
            height, width = frame.shape[:2]
        return VideoMeta(path, fps, max(n_frames, 0), width, height, reliable, True)
    finally:
        cap.release()


def frame_ok(frame: np.ndarray | None) -> bool:
    return frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0


class FrameReader:
    """Sequentially decode a video and yield every ``stride``-th frame.

    Decoding runs in a background thread (bounded queue) so the GPU never waits
    on the decoder.  ``stride`` may be raised while iterating (runtime
    governor); ordering is preserved, so results remain deterministic for a
    fixed stride schedule.
    """

    def __init__(self, path: str, fps: float, stride: int = 1, n_frames_hint: int = 0,
                 max_failures: int = 25, prefetch: int = 48):
        self.path = path
        self.fps = fps
        self.stride = max(1, int(stride))
        self.n_frames_hint = n_frames_hint
        self.max_failures = max_failures
        self.prefetch = max(2, prefetch)
        self.frames_decoded = 0     # number of frame positions advanced
        self.read_failures = 0
        self._stop = threading.Event()

    def _worker(self, q: queue.Queue) -> None:
        cap = cv2.VideoCapture(self.path)
        try:
            if not cap.isOpened():
                return
            idx = 0
            failures = 0
            while not self._stop.is_set():
                keep = idx % self.stride == 0
                if keep:
                    ok, frame = cap.read()
                    ok = ok and frame_ok(frame)
                else:
                    ok = cap.grab()
                    frame = None
                if not ok:
                    past_end = self.n_frames_hint <= 0 or idx >= self.n_frames_hint - 1
                    if past_end:
                        break  # normal end of stream
                    failures += 1
                    self.read_failures += 1
                    if failures > self.max_failures:
                        break
                    idx += 1
                    continue
                failures = 0
                self.frames_decoded = idx + 1
                if keep:
                    item = (idx, idx / self.fps, frame)
                    while not self._stop.is_set():
                        try:
                            q.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                idx += 1
        except Exception as exc:  # decoder crash must not kill the harness
            log.warning("%s: decoder error at frame %d: %s", self.path, self.frames_decoded, exc)
        finally:
            cap.release()
            while not self._stop.is_set():
                try:
                    q.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        thread = threading.Thread(target=self._worker, args=(q,), daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield item
        finally:
            self._stop.set()
            thread.join(timeout=5.0)

    def close(self) -> None:
        self._stop.set()


def effective_duration(meta: VideoMeta, frames_decoded: int) -> float:
    """Conservative video duration: never exceed what metadata or decoding saw."""
    decoded = frames_decoded / meta.fps if meta.fps > 0 else 0.0
    if meta.duration > 0 and decoded > 0:
        return min(meta.duration, decoded)
    return max(meta.duration, decoded)
