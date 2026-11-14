"""Romania - "Shaken, Still Standing": mixture of per-network experts.

    python solve.py [path/to/data]        (default: ./data)

1. route every record (train and sealed) to one of the three networks:
     B "quiet"  : the 36 heavy-tailed channels are attenuated about 4x (low RMS gain)
     A          : loud, c13 <= -0.59
     C          : loud, c13 >  -0.59
2. fit one expert per network on the train records of that network:
     A : 3-bit parity rule on c35, c43, c38 (random forest on those 3 channels)
     B : soft vote of quadratic logistic regression, RBF-SVM and QDA on the
         12 Gaussian channels (an ellipsoidal "energy" boundary)
     C : plain logistic regression on all 48 channels (a linear boundary)
3. report 5-fold CV accuracy per expert, write predictions.csv, plot figures,
   run the evaluator.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
C = [f"c{j + 1:02d}" for j in range(48)]
NAMES = {0: "B (quiet, 4x attenuated)", 1: "A (loud, c13<=-0.59)", 2: "C (loud, c13>-0.59)"}

tr = pd.read_csv(DATA / "dataset" / "train.csv")
se = pd.read_csv(DATA / "dataset" / "sealed.csv")
X, y, Xs = tr[C].to_numpy(), tr.vrancea.to_numpy(), se[C].to_numpy()
Z = np.vstack([X, Xs])

# 1. network routing ---------------------------------------------------------------
k = kurtosis(X, axis=0)
HEAVY = np.where(k > 0.5)[0]            # 36 scale-mixture channels
GAUSS = list(np.where(k < 0.4)[0])      # 12 Gaussian channels (never attenuated)
gain = np.sqrt(((Z[:, HEAVY] - np.median(Z[:, HEAVY], 0)) ** 2).mean(1))
c13 = Z[:, C.index("c13")]
net = np.where(gain < 0.6, 0, np.where(c13 <= -0.59, 1, 2))
nt, ns = net[:len(X)], net[len(X):]
print("Gaussian channels:", [C[j] for j in GAUSS])
for i in range(3):
    print(f"network {NAMES[i]:28s} train {np.sum(nt == i):5d}  sealed {np.sum(ns == i):5d}"
          f"  P(vrancea|train) {y[nt == i].mean():.3f}")

# 2. experts --------------------------------------------------------------------------
K3 = [C.index(c) for c in ("c35", "c43", "c38")]
ALL = list(range(48))
EXPERTS = {
    0: [("quad_lr", GAUSS, lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                                 StandardScaler(), LogisticRegression(C=1, max_iter=20000))),
        ("svm_rbf", GAUSS, lambda: make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0))),
        ("qda", GAUSS, lambda: QuadraticDiscriminantAnalysis())],
    1: [("rf_parity", K3, lambda: RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=0))],
    2: [("lr", ALL, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000)))],
}


def vote(models, Xq):
    return np.mean([m.predict_proba(Xq[:, cols])[:, 1] for (_, cols, _), m in models], 0)


# CV estimate (honest: routing is label-free, experts are refit inside each fold)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
cv_acc = {}
for i in range(3):
    idx = np.where(nt == i)[0]
    p = np.zeros(len(idx))
    single = {n: np.zeros(len(idx)) for n, _, _ in EXPERTS[i]}
    for a, b in cv.split(idx, y[idx]):
        fitted = [(e, e[2]().fit(X[idx[a]][:, e[1]], y[idx[a]])) for e in EXPERTS[i]]
        for (n, cols, _), m in fitted:
            single[n][b] = m.predict_proba(X[idx[b]][:, cols])[:, 1]
        p[b] = vote(fitted, X[idx[b]])
    for n, q in single.items():
        print(f"  CV {NAMES[i]:28s} {n:10s} {((q > .5) == y[idx]).mean():.4f}")
    cv_acc[i] = ((p > .5) == y[idx]).mean()
    print(f"  CV {NAMES[i]:28s} {'EXPERT':10s} {cv_acc[i]:.4f}")
est = sum(cv_acc[i] * np.mean(ns == i) for i in range(3))
print(f"expected sealed accuracy (CV weighted by sealed network mix): {est:.4f}  (need >= 0.9252)")

# parity rule check for network A
W = X[nt == 1]
rule = 1 ^ (W[:, C.index("c43")] > -0.615) ^ (W[:, C.index("c35")] > -0.225) ^ (W[:, C.index("c38")] > 0.73)
print(f"network A closed-form parity rule accuracy: {(rule.astype(int) == y[nt == 1]).mean():.4f}")

# 3. final fit + predict -----------------------------------------------------------
pred = np.zeros(len(Xs), int)
for i in range(3):
    m = nt == i
    fitted = [(e, e[2]().fit(X[m][:, e[1]], y[m])) for e in EXPERTS[i]]
    pred[ns == i] = (vote(fitted, Xs[ns == i]) > .5).astype(int)
out = pd.DataFrame({"record_id": se.record_id, "vrancea": pred})
out.to_csv(HERE / "predictions.csv", index=False)
print("wrote predictions.csv", out.vrancea.value_counts().to_dict())

# figures ----------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cols = ["#1f77b4", "#d62728", "#2ca02c"]
    for i in range(3):
        mm = net == i
        ax[0].scatter(np.log(gain[mm]), c13[mm], s=2, alpha=.4, color=cols[i], label=NAMES[i])
    ax[0].axvline(np.log(0.6), color="k", lw=.8, ls="--"); ax[0].axhline(-0.59, color="k", lw=.8, ls="--")
    ax[0].set_xlabel("log RMS gain of the 36 heavy-tailed channels"); ax[0].set_ylabel("c13")
    ax[0].set_title("The three networks (train + sealed)"); ax[0].legend(markerscale=6, fontsize=8)
    labels = ["global LR", "global SVM", "global HGB", "B expert", "A expert", "C expert", "MoE (sealed mix)"]
    vals = [0.601, 0.676, 0.847, cv_acc[0], cv_acc[1], cv_acc[2], est]
    ax[1].barh(labels, vals, color=["#999"] * 3 + cols + ["#000"])
    ax[1].axvline(0.9252, color="r", ls="--", label="pass threshold 92.52%")
    ax[1].set_xlim(0.5, 1); ax[1].set_xlabel("5-fold CV accuracy"); ax[1].legend(loc="lower right")
    for j, v in enumerate(vals):
        ax[1].text(v + .005, j, f"{v:.3f}", va="center", fontsize=8)
    ax[1].set_title("Models vs threshold")
    fig.tight_layout(); fig.savefig(HERE / "figure_networks_and_scores.png", dpi=130)
    print("wrote figure_networks_and_scores.png")
except ImportError:
    print("matplotlib missing: skipped figure")

print("\nevaluator says:")
sys.stdout.flush()
subprocess.run([sys.executable, str(DATA / "evaluation_script.py"), str(HERE / "predictions.csv")], check=False)
