"""NORWAY - The Nisse's Bad Shift.

Pipeline:
  1. Train a strong digit classifier (RBF-SVC) on the TRAIN images with their TRUE labels.
  2. For every sample in the target set compute, for each candidate swap pair (a,b), the evidence
     s_i = log p(other | x_i) - log p(given | x_i)  for samples whose given label is a or b.
  3. A bad night = a contiguous block of sample_ids in which EVERY a/b is swapped.  For each pair
     find the maximum-sum contiguous segment (Kadane over samples ordered by id).  Accept the best
     (pair, segment) if its evidence is overwhelming, flip every a/b label inside it, mask it, repeat.
  4. Retrain the classifier on train(true) + test(corrected) and save model.joblib.
Does NOT use the public load_digits lookup.
"""
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PIX = [f"pixel_{i}" for i in range(64)]
CLIP = 6.0
MIN_SCORE = 12.0     # total log-evidence needed to declare a night
MIN_SUPPORT = 4      # min number of samples in the segment that the classifier says are swapped


def make_clf():
    return SVC(C=10, gamma=0.001, probability=True, random_state=0)


def detect_nights(ids, given, proba, classes, min_score=MIN_SCORE, min_support=MIN_SUPPORT, max_nights=10):
    """ids: int array (ordering key), given: labels, proba: (n,10) class probabilities.
    Returns corrected labels and list of detected nights (pair, start_id, end_id, score, n_flipped)."""
    order = np.argsort(ids)
    ids_o, g_o, p_o = ids[order], given[order], np.clip(proba[order], 1e-6, 1)
    col = {c: k for k, c in enumerate(classes)}
    logp = np.log(p_o)
    masked = np.zeros(len(ids_o), bool)
    corrected = g_o.copy()
    nights = []
    for _ in range(max_nights):
        best = None
        for a in classes:
            for b in classes:
                if a >= b:
                    continue
                m = ((g_o == a) | (g_o == b)) & ~masked
                pos = np.where(m)[0]
                if len(pos) == 0:
                    continue
                other = np.where(g_o[pos] == a, b, a)
                s = logp[pos, [col[o] for o in other]] - logp[pos, [col[x] for x in g_o[pos]]]
                s = np.clip(s, -CLIP, CLIP)
                # Kadane, but do not let segments jump across masked (already-assigned) samples
                blk = np.concatenate([[0], np.cumsum(masked)])[pos]
                cur, cs, bs, bi, bj = 0.0, 0, -np.inf, 0, 0
                for k in range(len(pos)):
                    if k > 0 and blk[k] != blk[k - 1]:
                        cur = 0.0
                    if cur <= 0:
                        cur, cs = s[k], k
                    else:
                        cur += s[k]
                    if cur > bs:
                        bs, bi, bj = cur, cs, k
                seg = pos[bi:bj + 1]
                support = int((s[bi:bj + 1] > 0).sum())
                if best is None or bs > best[0]:
                    best = (bs, a, b, seg, support)
        if best is None or best[0] < min_score or best[4] < min_support:
            break
        bs, a, b, seg, support = best
        lo, hi = seg[0], seg[-1]
        for k in seg:
            corrected[k] = b if g_o[k] == a else a
        masked[lo:hi + 1] = True
        nights.append(dict(pair=(int(a), int(b)), start_id=int(ids_o[lo]), end_id=int(ids_o[hi]),
                           score=round(float(bs), 2), n_flipped=len(seg), support=support))
    out = np.empty_like(corrected)
    out[order] = corrected
    return out, nights


def idnum(s):
    return s.str.replace("IMG_", "", regex=False).astype(int).values


def main():
    tr = pd.read_csv(DATA / "train-input.csv").merge(pd.read_csv(DATA / "train-output.csv"), on="sample_id")
    te = pd.read_csv(DATA / "test-input.csv")
    Xtr, ytr = tr[PIX].values.astype(float), tr["corrected_label"].values
    Xte, gte = te[PIX].values.astype(float), te["given_label"].values

    clf = make_clf().fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    fixed, nights = detect_nights(idnum(te["sample_id"]), gte, proba, clf.classes_)
    for n in nights:
        print("night:", n)
    changed = int((fixed != gte).sum())
    print(f"labels changed: {changed} / {len(te)}")

    pd.DataFrame({"sample_id": te["sample_id"], "corrected_label": fixed.astype(int)}).to_csv(
        HERE / "test-output.csv", index=False)

    final = make_clf().fit(np.vstack([Xtr, Xte]), np.concatenate([ytr, fixed]))
    joblib.dump(final, HERE / "model.joblib")
    print("wrote test-output.csv and model.joblib")


if __name__ == "__main__":
    main()
