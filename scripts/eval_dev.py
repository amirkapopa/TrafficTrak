"""Score predictions against HUMAN-REVIEWED development labels.

    python scripts/eval_dev.py --pred predictions_samples.json --labels labels/dev_labels.json \
        [--risk-dir outputs/risk]

Reports per-class precision / recall / F1 at temporal IoU 0.3, 0.5, 0.7
(greedy one-to-one matching) and, if risk CSVs exist, ROC-AUC and Brier score
of the risk curve against "accident starts within the next 5 s".  This is a
local approximation; the organisers' evaluate.py is authoritative.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from _bootstrap import ROOT

from src.metrics import THRESHOLDS, brier, evaluate_events, risk_targets, roc_auc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=str(ROOT / "predictions_samples.json"))
    ap.add_argument("--labels", default=str(ROOT / "labels" / "dev_labels.json"))
    ap.add_argument("--risk-dir", default=str(ROOT / "outputs" / "risk"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dev_eval.json"))
    args = ap.parse_args()
    if not Path(args.labels).is_file():
        print(f"{args.labels} not found - create it with scripts/annotate_candidates.py (human review required)",
              file=sys.stderr)
        return 1
    gt = json.loads(Path(args.labels).read_text())
    pred = json.loads(Path(args.pred).read_text())
    rep = evaluate_events(pred, gt)
    print(f"{'class':22s}" + "".join(f"  F1@{t}" for t in THRESHOLDS))
    for cls, d in sorted(rep["per_class"].items()):
        print(f"{cls:22s}" + "".join(f"  {d.get(str(t), {}).get('f1', 0):.3f} " for t in THRESHOLDS))
    for t, d in rep["per_threshold"].items():
        print(f"tIoU {t}: P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} (tp {d['tp']}, fp {d['fp']}, fn {d['fn']})")
    print(f"mean class F1: {rep['mean_class_f1']:.3f}")
    ys, ss = [], []
    for vid, segs in gt.items():
        f = Path(args.risk_dir) / f"{Path(vid).stem}.csv"
        if not f.is_file():
            continue
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        t = np.array([float(r["t_sec"]) for r in rows])
        ys.append(risk_targets(t, [s[0] for s in segs if s[2] == "accident"]))
        ss.append(np.array([float(r["risk"]) for r in rows]))
    if ys:
        y, s = np.concatenate(ys), np.concatenate(ss)
        rep["risk"] = {"auc": roc_auc(y, s), "brier": brier(y, s), "positives": int(y.sum()), "frames": int(len(y))}
        print(f"risk: AUC={rep['risk']['auc']} Brier={rep['risk']['brier']:.4f} ({rep['risk']['positives']} positive frames)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
