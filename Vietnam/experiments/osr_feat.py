"""Open-set heads on frozen backbone features (extract_feats.py output). Small, CPU.
Protocols (all on REAL photos unless stated):
  CV   : 5-fold stratified; heads fit on the train folds (+ augmented views); known accuracy after rejection.
  LOBO : for each brand h, heads fit on the other two brands (train folds); unknowns = all real photos of h.
  ZAP  : Zappos PUMA/Reebok/Vans/ASICS (true test brands, catalogue domain) as unknowns; Zappos held-out
         Nike/adidas/Converse as domain-shifted knowns.
Threshold = quantile q of the held-out real known scores (fraction of real knowns rejected).
Usage: python osr_feat.py feats/mci0v2.npz"""
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
CLS = ["adidas", "converse", "nike"]
d = np.load(sys.argv[1])
l2 = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
R = l2(d["real"]); RA = l2(d["real_aug"]); y = np.array([CLS.index(x) for x in d["real_lab"]])
ZK = l2(d["zk_va"]); zky = np.array([CLS.index(x) for x in d["zk_va_lab"]]); ZU = l2(d["zu_va"]); zub = np.array([b.split()[0].lower() for b in d["zu_va_lab"]])
OE = l2(d["oe_tr"]); OEV = l2(d["oe_va"])
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(R, y))

def fit_heads(Xtr, ytr, classes, use_oe=False):
    """Returns dict name -> function(X) -> (score [higher=known], pred [index into classes])."""
    H = {}
    lr = LogisticRegression(C=10, max_iter=5000).fit(Xtr, ytr)
    H["msp"] = lambda X: (lr.predict_proba(X).max(1), lr.predict(X))
    H["maxlogit"] = lambda X: (lr.decision_function(X).max(1) if len(classes) > 2 else np.abs(lr.decision_function(X)), lr.predict(X))
    protos = l2(np.stack([Xtr[ytr == c].mean(0) for c in classes]))
    H["proto"] = lambda X: ((X @ protos.T).max(1), lr.predict(X))
    def knn(X, k=10): return np.sort(X @ Xtr.T, 1)[:, -k:].mean(1)
    H["knn"] = lambda X: (knn(X), lr.predict(X))
    # class-conditional kNN: similarity to the nearest neighbours OF THE PREDICTED CLASS
    def cknn(X, k=10):
        p = lr.predict(X); s = np.zeros(len(X))
        for c in classes:
            m = p == c
            if m.any(): s[m] = np.sort(X[m] @ Xtr[ytr == c].T, 1)[:, -k:].mean(1)
        return s
    H["cknn"] = lambda X: (cknn(X), lr.predict(X))
    mu = Xtr.mean(0); U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); P = Vt[:128].T
    Z = (Xtr - mu) @ P; cm = np.stack([Z[ytr == c].mean(0) for c in classes])
    lw = LedoitWolf().fit(Z - cm[np.searchsorted(classes, ytr)]); Pi = lw.precision_
    def maha(X):
        z = (X - mu) @ P; dd = np.stack([np.einsum("ij,jk,ik->i", z - m, Pi, z - m) for m in cm], 1)
        return -dd.min(1)
    H["maha"] = lambda X: (maha(X), lr.predict(X))
    # z-scored combinations (stats from training data scores)
    def zs(fn):
        s = fn(Xtr)[0]; m_, sd_ = s.mean(), s.std() + 1e-9
        return lambda X: (fn(X)[0] - m_) / sd_
    zk_, zm_, zl_ = zs(H["cknn"]), zs(H["maha"]), zs(H["maxlogit"])
    H["cknn+maha"] = lambda X: (zk_(X) + zm_(X), lr.predict(X))
    H["cknn+logit"] = lambda X: (zk_(X) + zl_(X), lr.predict(X))
    if use_oe:
        lr4 = LogisticRegression(C=10, max_iter=5000).fit(np.r_[Xtr, OE], np.r_[ytr, np.full(len(OE), 99)])
        H["oe4"] = lambda X: (1 - lr4.predict_proba(X)[:, -1], lr.predict(X))
    return H

def train_set(idx, keep):
    m = np.isin(y[idx], keep)
    Xtr = np.r_[R[idx][m], RA[:, idx][:, m].reshape(-1, R.shape[1])]
    ytr = np.r_[y[idx][m], np.tile(y[idx][m], RA.shape[0])]
    return Xtr, ytr

names = None; Q = [0.03, 0.05, 0.08, 0.10, 0.15]
# ---- CV (3 known classes) ----
cv = {}
for tr, va in folds:
    Xtr, ytr = train_set(tr, [0, 1, 2]); H = fit_heads(Xtr, ytr, [0, 1, 2], use_oe=True); names = list(H)
    for n, fn in H.items():
        s, p = fn(R[va]); sz, pz = fn(ZK); su, _ = fn(ZU); so, _ = fn(OEV)
        cv.setdefault(n, []).append((s, p == y[va], sz, pz == zky, su, so))
print("=== CV on real knowns (3 classes) + Zappos test brands as unknowns ===")
print(f"{'score':11s} closed_acc | AUROC(real_kn vs zap_unk) AUROC(zap_kn vs zap_unk) | " + " | ".join(f"q={q:.2f}: acc  zapU_rej zapK_acc" for q in Q))
for n in names:
    s = np.concatenate([c[0] for c in cv[n]]); ok = np.concatenate([c[1] for c in cv[n]])
    line = f"{n:11s} {ok.mean():.3f}      | "
    su = np.concatenate([c[4] for c in cv[n]]); sz = np.concatenate([c[2] for c in cv[n]]); okz = np.concatenate([c[3] for c in cv[n]])
    line += f"{roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(su))], np.r_[s, su]):.3f}                     {roc_auc_score(np.r_[np.ones(len(sz)), np.zeros(len(su))], np.r_[sz, su]):.3f}                  |"
    for q in Q:
        # per-fold threshold at quantile q of that fold's held-out known scores
        acc, rej, zacc = [], [], []
        for c in cv[n]:
            t = np.quantile(c[0], q); acc.append(np.mean(c[1] & (c[0] >= t))); rej.append(np.mean(c[4] < t)); zacc.append(np.mean(c[3] & (c[2] >= t)))
        line += f" {np.mean(acc):.3f} {np.mean(rej):.3f}    {np.mean(zacc):.3f}   |"
    print(line)
# ---- LOBO (2 known classes, real photos of the held-out brand = unknowns) ----
print("\n=== LOBO on real photos: rejection of the held-out brand (pooled over 3 brands x 5 folds) ===")
print(f"{'score':11s} AUROC  | " + " | ".join(f"q={q:.2f}: acc2  rej" for q in Q) + " | per-brand AUROC a/c/n")
lobo = {}
for h in range(3):
    keep = [c for c in range(3) if c != h]
    for tr, va in folds:
        Xtr, ytr = train_set(tr, keep); H = fit_heads(Xtr, ytr, keep)
        kv = va[y[va] != h]; un = np.where(y == h)[0]
        for n, fn in H.items():
            s, p = fn(R[kv]); su, _ = fn(R[un])
            lobo.setdefault(n, []).append((h, s, p == y[kv], su))
for n in lobo:
    line = f"{n:11s} "
    au = [roc_auc_score(np.r_[np.ones(len(c[1])), np.zeros(len(c[3]))], np.r_[c[1], c[3]]) for c in lobo[n]]
    line += f"{np.mean(au):.3f}  |"
    for q in Q:
        acc, rej = [], []
        for c in lobo[n]:
            t = np.quantile(c[1], q); acc.append(np.mean(c[2] & (c[1] >= t))); rej.append(np.mean(c[3] < t))
        line += f" {np.mean(acc):.3f} {np.mean(rej):.3f} |"
    pb = [np.mean([a for a, c in zip(au, lobo[n]) if c[0] == h]) for h in range(3)]
    print(line + " " + "/".join(f"{v:.3f}" for v in pb))
