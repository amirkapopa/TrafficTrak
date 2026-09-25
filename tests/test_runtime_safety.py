"""Guards that keep a video inside the harness time budget (3x duration for A + B)."""

import onnxruntime as ort
import pytest

from src import detection
from src.config import load_config, weights_dir
from src.risk import RiskEstimator

from .fake_detector import RectangleDetector, render

HAVE_BOTH = all((weights_dir() / f).is_file() for f in ("yolox_s.onnx", "yolox_m.onnx"))


@pytest.mark.skipif(not HAVE_BOTH, reason="needs both detector weights")
def test_cpu_fallback_switches_to_small_model(monkeypatch):
    """CUDA advertised but unusable -> session runs on CPU -> must not keep YOLOX-M."""
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(detection, "_DETECTORS", {})
    det = detection.get_detector(load_config())
    assert not det.on_gpu
    assert det.model_path.name == "yolox_s.onnx"


class SlowDetector(RectangleDetector):
    def __call__(self, frame):
        import time
        time.sleep(0.05)  # 50 ms per call = 1.25x real time at 25 fps
        return super().__call__(frame)


def test_risk_governor_reduces_detection_rate_when_too_slow():
    est = RiskEstimator(detector=SlowDetector(), overrides={"risk": {"detect_every_cpu": 1, "budget_factor": 0.5}})
    est.reset({"fps": 25.0, "width": 640, "height": 360})
    frame = render([(10, 10, 60, 40)])
    for i in range(200):
        est.step(frame, i / 25.0)
    assert est.detect_every > 1


def test_risk_governor_idle_when_fast():
    est = RiskEstimator(detector=RectangleDetector(), overrides={"risk": {"detect_every_cpu": 1}})
    est.reset({"fps": 25.0, "width": 640, "height": 360})
    frame = render([(10, 10, 60, 40)])
    for i in range(200):
        est.step(frame, i / 25.0)
    assert est.detect_every == 1
