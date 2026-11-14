"""Evaluate the three Delphi oracle models (Ares, Athena, Apollo).

Usage:  python evaluate.py [--data data]

- Loads each checkpoint with torch.load(..., weights_only=True) (untrusted .pth files).
- Uses the feature order defined in data/model.py (FEATURE_COLUMNS); region and
  moon_phase are NOT model inputs (the model is a single Linear(9, 1) layer on raw values).
- Threshold 0.50 on sigmoid(logit).
- Writes metrics.csv, metrics.md, confusion_matrices.png.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["ares", "athena", "apollo"]
THRESHOLD = 0.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()

    sys.path.insert(0, args.data)
    from model import FEATURE_COLUMNS, MythicThreatClassifier  # noqa: E402

    X_df = pd.read_csv(os.path.join(args.data, "test_dataset.csv"))
    y_df = pd.read_csv(os.path.join(args.data, "test_labels.csv"))
    df = X_df.merge(y_df, on="encounter_id", validate="one_to_one")
    assert len(df) == len(X_df) == len(y_df)
    X = torch.tensor(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    y = df["is_catastrophic_threat"].to_numpy().astype(int)
    print(f"Test set: {len(y)} rows, positives={y.sum()} ({y.mean():.1%})")

    rows, cms = [], {}
    for name in MODELS:
        model = MythicThreatClassifier()
        state = torch.load(os.path.join(args.data, "models", f"model_{name}.pth"),
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(X)).numpy()
        pred = (prob >= THRESHOLD).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cms[name] = cm
        rows.append({
            "model": name.capitalize(),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "f2": fbeta_score(y, pred, beta=2, zero_division=0),
            "roc_auc": roc_auc_score(y, prob),
            "predicted_pos": int(pred.sum()),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })

    m = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(m.round(4).to_string(index=False))
    m.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    with open(os.path.join(args.out, "metrics.md"), "w") as f:
        r4 = m.round(4).astype(str)
        f.write("| " + " | ".join(r4.columns) + " |\n")
        f.write("|" + "---|" * len(r4.columns) + "\n")
        for _, row in r4.iterrows():
            f.write("| " + " | ".join(row) + " |\n")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, name, r in zip(axes, MODELS, rows):
        cm = cms[name]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{r['model']}\nacc={r['accuracy']:.3f}  recall={r['recall']:.3f}  "
                     f"prec={r['precision']:.3f}", fontsize=10)
    fig.suptitle("Confusion matrices at threshold 0.50 (positive = catastrophic threat)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "confusion_matrices.png"), dpi=150)

    best = max(rows, key=lambda r: (r["recall"], r["f2"]))
    print(f"\nBest (max recall, tie-break F2): {best['model']}  ->  SIGMOID_{best['model'].upper()}")


if __name__ == "__main__":
    main()
