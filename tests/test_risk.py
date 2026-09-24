"""Part B: range, reset, causality, determinism and qualitative behaviour."""

import numpy as np

from src.risk import RiskEstimator

from .fake_detector import RectangleDetector, render

FPS = 15.0
META = {"video_id": "synthetic", "fps": FPS, "width": 640, "height": 360, "n_frames": 150}


def head_on(n=90):
    """Two cars approaching each other fast; B brakes hard ~1 s before impact (t~1.65 s)."""
    frames = []
    for i in range(n):
        t = i / FPS
        xa = 40 + 220 * t
        xb = 600 - 220 * t if t < 0.5 else 490 - 40 * (t - 0.5)
        frames.append(render([(xa, 200, xa + 40, 225), (xb, 200, xb + 40, 225)]))
    return frames


def parallel(n=90):
    frames = []
    for i in range(n):
        x = 20 + 150 * i / FPS
        frames.append(render([(x, 100, x + 40, 125), (x, 250, x + 40, 275)]))
    return frames


def run(frames, est=None):
    est = est or RiskEstimator(detector=RectangleDetector(), overrides={"risk": {"detect_every_cpu": 1}})
    est.reset(META)
    return [est.step(f, i / FPS) for i, f in enumerate(frames)], est


def test_outputs_are_probabilities():
    vals, _ = run(head_on())
    assert all(0.0 <= v <= 1.0 and np.isfinite(v) for v in vals)


def test_conflict_raises_risk_but_parallel_traffic_does_not():
    conflict, _ = run(head_on())
    calm, _ = run(parallel())
    first_alarm = next(i for i, v in enumerate(conflict) if v >= 0.5) / FPS
    assert first_alarm < 1.65  # raised before contact
    assert max(calm) < 0.1


def test_causality_future_frames_do_not_change_past_outputs():
    a = head_on(60)
    b = a[:30] + parallel(30)  # identical first 30 frames, different future
    va, _ = run(a)
    vb, _ = run(b)
    assert va[:30] == vb[:30]


def test_reset_clears_state_and_is_deterministic():
    est = RiskEstimator(detector=RectangleDetector(), overrides={"risk": {"detect_every_cpu": 1}})
    first, _ = run(head_on(), est)
    second, _ = run(head_on(), est)  # same instance, reset in between
    assert first == second
    est.reset(META)
    assert est.kin == {} and est.value == 0.0 and est.frame_no == 0 and est.tracker.next_id == 1


def test_bad_inputs_never_crash():
    est = RiskEstimator(detector=RectangleDetector())
    est.reset({})  # unknown fps / size
    for frame in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8), render([])):
        v = est.step(frame, 0.1)
        assert 0.0 <= v <= 1.0
    assert 0.0 <= est.step(render([]), float("nan")) <= 1.0


def test_solution_wrapper_contract():
    import solution

    est = solution.RiskEstimator()
    est._impl = RiskEstimator(detector=RectangleDetector())
    est.reset(META)
    assert 0.0 <= est.step(render([(10, 10, 50, 40)]), 0.0) <= 1.0
