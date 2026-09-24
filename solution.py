"""TrafficTrak submission entry point.

Part A: ``detect_events(video_path) -> [[start_sec, end_sec, label], ...]``
Part B: ``RiskEstimator`` with ``reset(meta)`` / ``step(frame, t_sec)``.

Everything runs offline with local open weights (``weights/``, fetched by
``bash weights/download.sh`` before evaluation).  All failures are logged to
stderr and degrade to empty / low-risk outputs instead of raising.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.logging_utils import setup_logging  # noqa: E402
from src.risk import RiskEstimator as _CausalRiskEstimator  # noqa: E402
from src.segments import CLASSES as _CLASSES  # noqa: E402

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
    "fire_smoke",
]
assert CLASSES == _CLASSES

_log = setup_logging()


def detect_events(video_path: str) -> list[list]:
    """Return [[start_sec, end_sec, label], ...]."""
    try:
        from src.pipeline import analyze_video
        from src.segments import validate_segments

        result = analyze_video(str(video_path))
        segments = result.segments()
        problems = validate_segments(segments, result.duration if result.duration > 0 else None)
        if problems:  # cannot happen by construction; never emit an invalid file
            _log.error("%s: dropping invalid output: %s", video_path, problems[:3])
            return []
        for w in result.warnings:
            _log.warning("%s: %s", video_path, w)
        return segments
    except Exception as exc:  # noqa: BLE001 - the harness must never crash
        _log.error("detect_events(%s) failed: %s", video_path, exc, exc_info=True)
        return []


class RiskEstimator:
    def __init__(self) -> None:
        self._impl = _CausalRiskEstimator()

    def reset(self, meta: dict) -> None:
        """
        meta includes video_id, fps, width, height, n_frames.
        Reset all tracker and temporal state.
        """
        try:
            self._impl.reset(meta)
        except Exception as exc:  # noqa: BLE001
            _log.error("RiskEstimator.reset failed: %s", exc, exc_info=True)
            self._impl = _CausalRiskEstimator()

    def step(self, frame, t_sec: float) -> float:
        """
        frame is BGR uint8 OpenCV image.
        Return P(accident starts within next 5 seconds) in [0, 1].
        Must be causal: only use this frame and earlier frames.
        Must not open/read the video file or reuse future Part A output.
        """
        value = self._impl.step(frame, t_sec)
        return float(min(max(value, 0.0), 1.0)) if value == value else 0.0


if __name__ == "__main__":
    import json

    for path in sys.argv[1:]:
        print(json.dumps({os.path.basename(path): detect_events(path)}))
