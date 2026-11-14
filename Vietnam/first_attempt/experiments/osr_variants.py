"""Head variants for the chosen score  s = z(class-conditional kNN) + z(max logit)  on frozen features.
Variants: kNN bank with/without augmented views, + Zappos knowns (catalogue domain), PCA dimension.
Reports CV (real) known accuracy, Zappos test-brand rejection, Zappos-known accuracy and real LOBO rejection
at fixed known-rejection quantiles q. Usage: python osr_variants.py feats.npz"""
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
CLS = ["adidas", "converse", "nike"]
d = np.load(sys.argv[1]); l2 = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
R = l2(d["real"]); RA = l2(d["real_aug"]); y = np.array([CLS.index(x) for x in d["real_lab"]])
ZKT = l2(d["zk_tr"]); zkty = np.array([CLS.index(x) for x in d["zk_tr_lab"]])
ZK = l2(d["zk_va"]); zky = np.array([CLS.index(x) for x in d["zk_va_lab"]]); ZU = l2(d["zu_va"])
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(R, y))
K = 10

class Head:
    def __init__(self, Xlr, ylr, Xb, yb, classes, pca):
        self.mu = Xlr.mean(0)
        self.P = np.linalg.svd(Xlr - self.mu, full_matrices=False)[2][:pca].T if pca else None
        f = self.f
        self.lr = LogisticRegression(C=10, max_iter=5000).fit(f(Xlr), ylr)
        self.bank = {c: f(Xb[yb == c]) for c in classes}; self.classes = classes
        st = lambda a: (a.mean(), a.std() + 1e-9)
        sk, sl = self.raw(Xlr); self.zk, self.zl = st(sk), st(sl)
    def f(self, X): return l2((X - self.mu) @ self.P) if self.P is not None else X
    def raw(self, X):
        Z = self.f(X); p = self.lr.predict(Z); sk = np.zeros(len(X))
        for c in self.classes:
            m = p == c
            if m.any(): sk[m] = np.sort(Z[m] @ self.bank[c].T, 1)[:, -K:].mean(1)
        lg = self.lr.decision_function(Z); sl = lg.max(1) if lg.ndim > 1 else np.abs(lg)
        return sk, sl
    def __call__(self, X):
        sk, sl = self.raw(X); Z = self.f(X)
        return (sk - self.zk[0]) / self.zk[1] + (sl - self.zl[0]) / self.zl[1], self.lr.predict(Z)

from sklearn.cluster import KMeans
def build(idx, keep, bank_aug, use_zk, pca, kmeans=0):
    m = np.isin(y[idx], keep)
    Xlr = np.r_[R[idx][m], RA[:, idx][:, m].reshape(-1, R.shape[1])]; ylr = np.r_[y[idx][m], np.tile(y[idx][m], RA.shape[0])]
    na = int(bank_aug)
    Xb = np.r_[R[idx][m], RA[:na, idx][:, m].reshape(-1, R.shape[1])]; yb = np.r_[y[idx][m], np.tile(y[idx][m], na)]
    if kmeans:
        Xk, yk = [], []
        for c in keep:
            km = KMeans(kmeans, n_init=1, random_state=0).fit(Xb[yb == c]); Xk.append(l2(km.cluster_centers_)); yk += [c] * kmeans
        Xb, yb = np.concatenate(Xk), np.array(yk)
    if use_zk:
        mz = np.isin(zkty, keep); Xlr = np.r_[Xlr, ZKT[mz]]; ylr = np.r_[ylr, zkty[mz]]; Xb = np.r_[Xb, ZKT[mz]]; yb = np.r_[yb, zkty[mz]]
    return Head(Xlr, ylr, Xb, yb, keep, pca)
VARIANTS = [("bank=real+4aug", 4, False, 0, 0),
            ("kmeans400/cls of real+4aug", 4, False, 0, 400)]

Q = [0.03, 0.05, 0.08, 0.10]
print("variant                       | " + " | ".join(f"q={q:.2f}: realAcc zapUrej zapKacc LOBOrej" for q in Q) + " | closed bank_n")
for name, ba, zk, pca, km in VARIANTS:
    cv = []; lb = []
    for tr, va in folds:
        H = build(tr, [0, 1, 2], ba, zk, pca, km)
        s, p = H(R[va]); sz, pz = H(ZK); su, _ = H(ZU)
        cv.append((s, p == y[va], su, sz, pz == zky)); nb = sum(len(b) for b in H.bank.values())
        for h in range(3):
            keep = [c for c in range(3) if c != h]; Hh = build(tr, keep, ba, zk, pca, km)
            kv = va[y[va] != h]; sk, _ = Hh(R[kv]); sh, _ = Hh(R[y == h]); lb.append((sk, sh))
    line = f"{name:29s} |"
    for q in Q:
        a, r, za, lr_ = [], [], [], []
        for (s, ok, su, sz, okz) in cv:
            t = np.quantile(s, q); a.append(np.mean(ok & (s >= t))); r.append(np.mean(su < t)); za.append(np.mean(okz & (sz >= t)))
        for sk, sh in lb:
            t = np.quantile(sk, q); lr_.append(np.mean(sh < t))
        line += f" {np.mean(a):.3f}   {np.mean(r):.3f}   {np.mean(za):.3f}   {np.mean(lr_):.3f}  |"
    print(line + f" {np.mean(np.concatenate([c[1] for c in cv])):.3f} {nb}", flush=True)
