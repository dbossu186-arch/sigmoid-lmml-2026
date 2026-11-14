"""Threshold analysis of one or several runs' val_logits.npz (CV folds are pooled).
Usage: python analyze.py [--score unk|msp|maxlogit] runs/a0 runs/a1 ...
The threshold is set as a quantile of the real held-out known scores (= fraction of real knowns rejected)."""
import sys, numpy as np
args = sys.argv[1:]
score = "auto"
if args and args[0] == "--score": score, args = args[1], args[2:]
def sm(l): e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def S(l, nk):
    p = sm(l)
    if score == "maxlogit": return l[:, :nk].max(1)
    if score == "msp" or l.shape[1] == nk: return p[:, :nk].max(1)
    return 1 - p[:, nk]  # explicit unknown class
pool = {}
for r in args:
    d = np.load(f"{r}/val_logits.npz"); known = list(d["known"]); nk = len(known)
    for k in ["real_va", "real_ho", "zk_va", "zu_va", "oe_va"]:
        l = d[k]; lab = d[k + "_lab"]
        if len(l) == 0: continue
        pool.setdefault(k, []).append((S(l, nk), l[:, :nk].argmax(1), np.array([known.index(x) if x in known else -1 for x in lab]), lab))
cat = {k: [np.concatenate([v[i] for v in vs]) for i in range(4)] for k, vs in pool.items()}
s, pr, y, _ = cat["real_va"]
kn = y >= 0
print(f"runs={len(args)} real_va n={kn.sum()} closed-set acc={np.mean(pr[kn]==y[kn]):.3f} | zk_va closed acc={np.mean(cat['zk_va'][1][cat['zk_va'][2]>=0]==cat['zk_va'][2][cat['zk_va'][2]>=0]):.3f}")
try:
    from sklearn.metrics import roc_auc_score
    su = cat["zu_va"][0]; zk = cat["zk_va"]; m = zk[2] >= 0
    print(f"AUROC zk_va vs zu_va={roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(len(su))], np.r_[zk[0][m], su]):.3f}  real_va vs zu_va={roc_auc_score(np.r_[np.ones(kn.sum()), np.zeros(len(su))], np.r_[s[kn], su]):.3f}", end="")
    if "real_ho" in cat: print(f"  real_va vs real_ho={roc_auc_score(np.r_[np.ones(kn.sum()), np.zeros(len(cat['real_ho'][0]))], np.r_[s[kn], cat['real_ho'][0]]):.3f}", end="")
    print()
except Exception as e: print(e)
print(" q    thr   | real_acc real_rej | zk_acc | zu_rej [puma reebok vans asics] | oe_rej | lobo_real_rej lobo_zk_rej")
for q in [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]:
    t = np.quantile(s[kn], q) if q > 0 else -np.inf
    acc = np.mean((pr[kn] == y[kn]) & (s[kn] >= t)); rej = np.mean(s[kn] < t)
    zs, zp, zy, _ = cat["zk_va"]; m = zy >= 0
    zacc = np.mean((zp[m] == zy[m]) & (zs[m] >= t)); zho = np.mean(zs[~m] < t) if (~m).any() else np.nan
    us, _, _, ub = cat["zu_va"]; ub = np.array([b.split()[0].lower() for b in ub])
    per = [np.mean(us[np.char.find(ub, b) >= 0] < t) for b in ["puma", "reebok", "vans", "asics"]]
    per[3] = np.mean(us[(ub == "asics") | (ub == "onitsuka")] < t)
    oe = np.mean(cat["oe_va"][0] < t)
    ho = np.mean(cat["real_ho"][0] < t) if "real_ho" in cat else np.nan
    print(f"{q:.2f} {t:7.3f} | {acc:.3f}   {rej:.3f}   | {zacc:.3f}  | {np.mean(us<t):.3f} [{' '.join(f'{v:.2f}' for v in per)}] | {oe:.3f} | {ho:.3f} {zho:.3f}")
