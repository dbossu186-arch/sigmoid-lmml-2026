import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
d = np.load(sys.argv[1]); B = list(d["brands"]); ls = float(d["logit_scale"])
X, y = d["real"], d["real_y"]; ZK, zky, ZU, ZO = d["zk"], d["zk_y"], d["zu"], d["zo"]
zub = np.array([b.split()[0].upper() for b in d["zu_brand"]])
TB, TE = d["text_brand"], d["text_extra"]
def zs(F_, neg_extra=True, neg_brands=True, drop=()):
    T = [TB[:3]]
    if neg_brands: T.append(np.stack([TB[i] for i in range(3, len(B)) if B[i] not in drop]))
    if neg_extra: T.append(TE)
    T = np.concatenate(T); L = ls * F_ @ T.T
    P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return P[:, :3].sum(1), P[:, :3].argmax(1), L
print("logit scale", ls)
for name, kw in [("all", {}), ("brands only", dict(neg_extra=False)), ("no test brands", dict(drop=("Puma","Reebok","Vans","Asics","Onitsuka Tiger")))]:
    pk_r, pr_r, _ = zs(X, **kw); pk_k, pr_k, _ = zs(ZK, **kw); pk_u, _, _ = zs(ZU, **kw); pk_o, _, _ = zs(ZO, **kw)
    print(f"== zero-shot [{name}] brand acc real={np.mean(pr_r==y):.3f} zk={np.mean(pr_k==zky):.3f} | AUC real-vs-zu={roc_auc_score(np.r_[np.ones(len(X)),np.zeros(len(ZU))], np.r_[pk_r,pk_u]):.3f} zk-vs-zu={roc_auc_score(np.r_[np.ones(len(ZK)),np.zeros(len(ZU))], np.r_[pk_k,pk_u]):.3f}")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ka = np.mean((pr_r == y) & (pk_r >= t)); kz = np.mean((pr_k == zky) & (pk_k >= t))
        rej = np.mean(pk_u < t); per = {b: round(float(np.mean(pk_u[zub == b] < t)), 2) for b in sorted(set(zub))}
        print(f"  t={t:.1f} knownacc real={ka:.3f} zk={kz:.3f} | rej zu={rej:.3f} {per} zo={np.mean(pk_o<t):.3f}")
# supervised: logistic on image emb, CV on real
skf = StratifiedKFold(5, shuffle=True, random_state=0)
clf = LogisticRegression(C=10, max_iter=3000)
pcv = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
clf.fit(X, y); pk = clf.predict_proba(ZK)
print(f"logreg brand acc cv-real={np.mean(pcv.argmax(1)==y):.3f} zk={np.mean(pk.argmax(1)==zky):.3f}")
# combined brand: zero-shot logits + logreg
_, _, Lr = zs(X); _, _, Lk = zs(ZK)
for a in [0.5, 1, 2]:
    sr = np.log(pcv + 1e-9) * a + (Lr[:, :3] - Lr[:, :3].max(1, keepdims=True)); sk = np.log(pk + 1e-9) * a + (Lk[:, :3])
    print(f"  combo a={a} acc cv-real={np.mean(sr.argmax(1)==y):.3f} zk={np.mean(sk.argmax(1)==zky):.3f}")
# kNN rejection on image space
def knn(Q, R, k=5): S = Q @ R.T; return np.sort(S, 1)[:, -k:].mean(1)
kr = np.zeros(len(X))
for tri, tei in skf.split(X, y): kr[tei] = knn(X[tei], X[tri])
ku = knn(ZU, X); kk = knn(ZK, X)
print(f"kNN AUC realcv-vs-zu={roc_auc_score(np.r_[np.ones(len(X)),np.zeros(len(ZU))], np.r_[kr,ku]):.3f} zk-vs-zu={roc_auc_score(np.r_[np.ones(len(ZK)),np.zeros(len(ZU))], np.r_[kk,ku]):.3f}")
# LOBO for kNN and for zero-shot (does holding out a brand => rejected?)
for ho in range(3):
    m = y != ho
    s_in = np.zeros(m.sum()); Xi, yi = X[m], y[m]
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=0).split(Xi, yi): s_in[tei] = knn(Xi[tei], Xi[tri])
    s_out = knn(X[~m], Xi)
    print(f"  LOBO ho={ho} kNN AUC={roc_auc_score(np.r_[np.ones(len(s_in)),np.zeros(len(s_out))], np.r_[s_in,s_out]):.3f}")
