"""Compare open-set scores on runs that saved features (val_logits.npz with *_feat).
Scores (higher = more known):
  unk   : 1 - p(unknown) from the 4-way head
  msp   : max known softmax prob
  maha  : -min class Mahalanobis distance (shared cov, fit on real train knowns, PCA-whitened)
  knn   : mean cosine sim to 5 nearest real train knowns
  combo : rank-free combination  z(unk_logit_margin) + z(maha)   (z-scored on real train)
Threshold = quantile q of the real held-out known scores. Pools several runs (e.g. 3 LOBO runs).
Usage: python analyze2.py runs/lobo_a runs/lobo_c runs/lobo_n"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
def sm(l): e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def l2(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
def scores(d):
    known = list(d["known"]); nk = len(known)
    ftr = l2(d["real_tr_feat"]); ytr = np.array([known.index(x) for x in d["real_tr_lab"]])
    mu = ftr.mean(0); U, S, Vt = np.linalg.svd(ftr - mu, full_matrices=False); P = Vt[:64].T / (S[:64] / np.sqrt(len(ftr)) + 1e-3)
    ztr = (ftr - mu) @ P; cm = np.stack([ztr[ytr == c].mean(0) for c in range(nk)])
    def S_(key):
        l = d[key]; f = l2(d[key + "_feat"]); p = sm(l)
        out = {"unk": 1 - p[:, -1] if l.shape[1] > nk else p.max(1), "msp": p[:, :nk].max(1) / p[:, :nk].sum(1)}
        z = (f - mu) @ P; out["maha"] = -np.min(((z[:, None] - cm[None]) ** 2).sum(-1), 1)
        out["knn"] = np.sort(f @ ftr.T, 1)[:, -5:].mean(1)
        out["unklogit"] = (l[:, :nk].max(1) - l[:, -1]) if l.shape[1] > nk else l.max(1)
        return out, l[:, :nk].argmax(1)
    res = {k: S_(k) for k in ["real_va", "real_ho", "zk_va", "zu_va", "oe_va", "real_tr"] if len(d[k])}
    # combo: z-score each component on real train knowns
    tr = res["real_tr"][0]
    st = {k: (tr[k].mean(), tr[k].std() + 1e-8) for k in ["unklogit", "maha", "knn"]}
    for k, (sc, _) in res.items():
        sc["combo"] = (sc["unklogit"] - st["unklogit"][0]) / st["unklogit"][1] + (sc["maha"] - st["maha"][0]) / st["maha"][1]
        sc["combo_knn"] = (sc["unklogit"] - st["unklogit"][0]) / st["unklogit"][1] + (sc["knn"] - st["knn"][0]) / st["knn"][1]
    lab = {k: np.array([known.index(x) if x in known else -1 for x in d[k + "_lab"]]) for k in res}
    return res, lab
pooled = {}
for r in sys.argv[1:]:
    d = np.load(f"{r}/val_logits.npz"); res, lab = scores(d)
    for k in res:
        for sn, v in res[k][0].items():
            pooled.setdefault((k, sn), []).append(v)
        pooled.setdefault((k, "pred"), []).append(res[k][1]); pooled.setdefault((k, "lab"), []).append(lab[k])
P = {k: np.concatenate(v) for k, v in pooled.items()}
has_ho = ("real_ho", "unk") in P
print(f"runs={len(sys.argv)-1}  (real_ho = real photos of the held-out brand)" if has_ho else f"runs={len(sys.argv)-1}")
print("score      | AUROC real_va~real_ho  real_va~zu | q=0.05: real_acc ho_rej zu_rej oe_rej | q=0.10: real_acc ho_rej zu_rej oe_rej | q=0.20: real_acc ho_rej zu_rej")
for sn in ["unk", "msp", "maha", "knn", "combo", "combo_knn"]:
    y = P[("real_va", "lab")]; kn = y >= 0; s = P[("real_va", sn)]; pr = P[("real_va", "pred")]
    # held-out-brand images inside real_va count as real unknowns too
    ho = np.r_[P[("real_ho", sn)]] if has_ho else s[~kn]
    zu = P[("zu_va", sn)]; oe = P[("oe_va", sn)]
    a1 = roc_auc_score(np.r_[np.ones(kn.sum()), np.zeros(len(ho))], np.r_[s[kn], ho]) if len(ho) else np.nan
    a2 = roc_auc_score(np.r_[np.ones(kn.sum()), np.zeros(len(zu))], np.r_[s[kn], zu])
    line = f"{sn:10s} | {a1:.3f}                 {a2:.3f}     |"
    for q in [0.05, 0.10, 0.20]:
        t = np.quantile(s[kn], q)
        acc = np.mean((pr[kn] == y[kn]) & (s[kn] >= t))
        line += f" {acc:.3f}   {np.mean(ho < t) if len(ho) else np.nan:.3f}  {np.mean(zu < t):.3f}" + (f"  {np.mean(oe < t):.3f} |" if q < 0.2 else " ")
    print(line)
