import numpy as np

from src.metrics import evaluate_events, match, risk_targets, roc_auc, tiou


def test_tiou_and_matching():
    assert tiou((0, 10), (5, 15)) == 5 / 15
    assert match([(0, 10), (20, 30)], [(1, 10)], 0.5) == (1, 1, 0)
    assert match([(0, 10)], [(8, 20)], 0.3) == (0, 1, 1)


def test_evaluate_events_report():
    gt = {"a.mp4": [[0, 10, "wrong_way"], [5, 30, "congestion"]]}
    pred = {"a.mp4": [[1, 10, "wrong_way"], [50, 60, "accident"]]}
    rep = evaluate_events(pred, gt)
    assert rep["per_class"]["wrong_way"]["0.7"]["f1"] == 1.0
    assert rep["per_class"]["accident"]["0.3"]["fp"] == 1
    assert rep["per_threshold"]["0.5"]["tp"] == 1


def test_risk_targets_and_auc():
    t = np.arange(0, 20, 1.0)
    y = risk_targets(t, [10.0])
    assert y.tolist() == [0] * 5 + [1] * 5 + [0] * 10
    assert roc_auc(y, y.astype(float)) == 1.0
    assert roc_auc(np.zeros(3), np.zeros(3)) is None
