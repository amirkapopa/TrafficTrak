"""Local development metrics.

IMPORTANT: the official ``evaluate.py`` was not available when this module
was written.  This is an approximation for tuning on *human-annotated* sample
clips only (``labels/dev_labels.json``): per-class greedy one-to-one matching
by temporal IoU at thresholds 0.3 / 0.5 / 0.7, reported as precision, recall
and F1, plus ROC-AUC / Brier score for the risk curve.  The official
evaluator is authoritative.
"""

from __future__ import annotations

import numpy as np

from .segments import CLASSES

THRESHOLDS = (0.3, 0.5, 0.7)


def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(pred: list, gt: list, thr: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching by descending IoU. Returns (tp, fp, fn)."""
    pairs = sorted(((tiou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g = set(), set()
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
    tp = len(used_p)
    return tp, len(pred) - tp, len(gt) - tp


def evaluate_events(pred: dict, gt: dict) -> dict:
    """pred / gt: {video: [[s, e, label], ...]} (only videos present in gt are scored)."""
    report = {"per_class": {}, "per_threshold": {}}
    for thr in THRESHOLDS:
        tot = np.zeros(3, dtype=np.int64)
        for label in CLASSES:
            c = np.zeros(3, dtype=np.int64)
            for vid, gsegs in gt.items():
                g = [(s[0], s[1]) for s in gsegs if s[2] == label]
                p = [(s[0], s[1]) for s in pred.get(vid, []) if s[2] == label]
                c += match(p, g, thr)
            tot += c
            if c.sum() == 0:
                continue
            tp, fp, fn = c.tolist()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            report["per_class"].setdefault(label, {})[str(thr)] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec,
                                                                     "recall": rec, "f1": f1}
        tp, fp, fn = tot.tolist()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        report["per_threshold"][str(thr)] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
                                             "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    f1s = [v["f1"] for cls in report["per_class"].values() for v in cls.values()]
    report["mean_class_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return report


def risk_targets(times: np.ndarray, accident_starts: list[float], horizon: float = 5.0) -> np.ndarray:
    """1 where an accident starts within (t, t + horizon]."""
    y = np.zeros(len(times), dtype=np.int64)
    for s in accident_starts:
        y |= ((s > times) & (s <= times + horizon)).astype(np.int64)
    return y


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):  # average ranks for ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def brier(y: np.ndarray, score: np.ndarray) -> float:
    return float(np.mean((np.asarray(score, dtype=np.float64) - np.asarray(y)) ** 2)) if len(y) else 0.0
