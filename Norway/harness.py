"""Local harness for NORWAY.
  1. repair_f1(): reproduces the platform metric (F1 over repairs).
  2. Held-out A: detect the real train night with train labels hidden (5-fold out-of-fold probabilities,
     classifier trained on true labels of the other folds only).
  3. Held-out B: simulated swap nights with random pairs/positions/counts on a held-out half of train,
     classifier trained on the other half only.
  4. Optional reference: score test-output.csv against public load_digits (only if --reference given;
     NOT used by main.py).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from main import make_clf, detect_nights, idnum, PIX, DATA, HERE


def repair_f1(given, pred, true):
    given, pred, true = map(np.asarray, (given, pred, true))
    changed = pred != given
    wrong = given != true
    tp = int((changed & (pred == true)).sum())
    fp = int((changed & (pred != true)).sum())
    fn = int((wrong & ~(changed & (pred == true))).sum())
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, p, r, tp, fp, fn


def heldout_train():
    tr = pd.read_csv(DATA / "train-input.csv").merge(pd.read_csv(DATA / "train-output.csv"), on="sample_id")
    X, y, g = tr[PIX].values.astype(float), tr["corrected_label"].values, tr["given_label"].values
    proba = np.zeros((len(tr), 10))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        proba[b] = make_clf().fit(X[a], y[a]).predict_proba(X[b])
    fixed, nights = detect_nights(idnum(tr["sample_id"]), g, proba, np.arange(10))
    print("[A] train night detection:", nights)
    print("[A] F1=%.4f P=%.4f R=%.4f tp=%d fp=%d fn=%d" % repair_f1(g, fixed, y))


def simulate(n_trials=60, seed=0):
    rng = np.random.default_rng(seed)
    tr = pd.read_csv(DATA / "train-input.csv").merge(pd.read_csv(DATA / "train-output.csv"), on="sample_id")
    X, y = tr[PIX].values.astype(float), tr["corrected_label"].values
    f1s = []
    for t in range(n_trials):
        perm = rng.permutation(len(X))
        fit, ev = perm[:600], perm[600:]
        clf = make_clf().fit(X[fit], y[fit])
        Xe, ye = X[ev], y[ev]
        ids = np.arange(len(ev))  # evaluation set in a random "night" order
        g = ye.copy()
        k = rng.integers(1, 4)  # 1-3 bad nights
        starts = np.sort(rng.choice(np.arange(0, len(ev) - 160, 1), k, replace=False))
        last = -1
        for s0 in starts:
            s0 = max(s0, last + 1)
            ln = int(rng.integers(60, 160))
            a, b = rng.choice(10, 2, replace=False)
            seg = slice(s0, min(s0 + ln, len(ev)))
            gg = g[seg]
            sw = np.where(ye[seg] == a, b, np.where(ye[seg] == b, a, gg))
            g[seg] = sw
            last = seg.stop
        fixed, nights = detect_nights(ids, g, clf.predict_proba(Xe), clf.classes_)
        f1 = repair_f1(g, fixed, ye)
        f1s.append(f1[0])
        if f1[0] < 0.95:
            print("  trial", t, "nights", k, "F1=%.3f P=%.3f R=%.3f tp=%d fp=%d fn=%d" % f1)
    f1s = np.array(f1s)
    print("[B] simulated nights: mean F1=%.4f  min=%.4f  P(F1>=0.93)=%.3f over %d trials"
          % (f1s.mean(), f1s.min(), (f1s >= 0.93).mean(), len(f1s)))


def reference():
    from sklearn.datasets import load_digits
    d = load_digits()
    key = {tuple(r): t for r, t in zip(d.data.astype(int), d.target)}
    te = pd.read_csv(DATA / "test-input.csv")
    out = pd.read_csv(HERE / "test-output.csv")
    assert (out.sample_id.values == te.sample_id.values).all()
    true = np.array([key[tuple(r)] for r in te[PIX].values.astype(int)])
    print("[REF] test vs public load_digits: F1=%.4f P=%.4f R=%.4f tp=%d fp=%d fn=%d"
          % repair_f1(te.given_label, out.corrected_label, true))


if __name__ == "__main__":
    heldout_train()
    simulate(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60)
    if "--reference" in sys.argv:
        reference()
