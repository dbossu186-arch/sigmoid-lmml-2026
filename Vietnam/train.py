"""Train the open-set head on top of the frozen, compressed MobileCLIP2-S0 image backbone.

Pipeline (see README.md):
  1. backbone.pt  = compressed + traced MobileCLIP2-S0 image tower (built by `export.py backbone`, never trained here)
  2. features     = L2-normalised backbone embeddings (flip-TTA) of the 735 real photos + N augmented views each
  3. head         = logistic regression (3 brands) + per-class kNN bank (k-means centroids of real+aug features)
  4. score        s = z(mean top-k cosine to the bank of the predicted class) + z(max LR logit)      (higher = known)
  5. threshold    = quantile q of the out-of-fold (5-fold CV) known scores   ->  reject if s < thr
Also reports the proxies: CV known accuracy, leave-one-brand-out rejection on real photos, and (if ext/ exists)
UT-Zappos50K PUMA/Reebok/Vans/ASICS rejection. Seeds are fixed; output head.npz is consumed by `export.py final`.

Usage (GPU box):  python train.py --backbone backbone.pt --out head.npz [--q 0.05]
"""
import argparse, csv, json, os, random
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "2")
import numpy as np, torch
from PIL import Image
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

p = argparse.ArgumentParser()
p.add_argument("--backbone", default="backbone.pt")
p.add_argument("--data", default="data")
p.add_argument("--out", default="head.npz")
p.add_argument("--n_aug", type=int, default=4, help="augmented views per real photo")
p.add_argument("--kmeans", type=int, default=400, help="bank centroids per class (0 = keep all vectors)")
p.add_argument("--k", type=int, default=10, help="kNN neighbours")
p.add_argument("--C", type=float, default=10.0)
p.add_argument("--q", type=float, default=0.05, help="fraction of held-out real knowns rejected (sets the threshold)")
p.add_argument("--feats_cache", default="feats_cache.npz")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(2)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CLS = ["adidas", "converse", "nike"]
l2 = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)

# ---------------- features ----------------
S = 256
plain = lambda im: T.functional.to_tensor(im.convert("RGB").resize((S, S), Image.BICUBIC))
aug = T.Compose([T.RandomResizedCrop(S, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(),
                 T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.6), T.ToTensor()])
rows = list(csv.DictReader(open(f"{args.data}/train-output.csv")))
paths = [f"{args.data}/train-input/{r['filename']}" for r in rows]
y = np.array([CLS.index(r["label"]) for r in rows])

def embed(bb, ps, tf, tta):
    out = []
    with torch.no_grad():
        for i in range(0, len(ps), 16):
            x = torch.stack([tf(Image.open(q)) for q in ps[i:i + 16]]).to(dev)
            f = bb(x)
            if tta: f = (f + bb(torch.flip(x, [3]))) / 2
            out.append(f.float().cpu())
    return l2(torch.cat(out).numpy())

if os.path.exists(args.feats_cache):
    c = np.load(args.feats_cache, allow_pickle=True); R, RA, extra = c["R"], c["RA"], c["extra"].item()
else:
    bb = torch.jit.load(args.backbone, map_location=dev).eval()
    R = embed(bb, paths, plain, True)
    RA = np.stack([embed(bb, paths, aug, False) for _ in range(args.n_aug)])
    extra = {}
    if os.path.exists("ext/zappos_index.csv"):  # optional proxy: UT-Zappos50K (validation only, never used for fitting)
        zi = list(csv.DictReader(open("ext/zappos_index.csv")))
        for nm, role, split in [("zap_unk", "val_unk", None), ("zap_known", "val_known", "val")]:
            sel = [r for r in zi if r["role"] == role and (split is None or r["split"] == split)]
            extra[nm] = embed(bb, [r["path"] for r in sel], plain, True)
            extra[nm + "_lab"] = np.array([r["label"] if role == "val_known" else r["brand"] for r in sel])
    np.savez(args.feats_cache, R=R, RA=RA, extra=np.array(extra, dtype=object))
print(f"features: real {R.shape}, aug {RA.shape}, extra {list(extra)}", flush=True)

# ---------------- head ----------------
class Head:
    def __init__(self, idx, keep):
        m = np.isin(y[idx], keep)
        Xlr = np.r_[R[idx][m], RA[:, idx][:, m].reshape(-1, R.shape[1])]
        ylr = np.r_[y[idx][m], np.tile(y[idx][m], RA.shape[0])]
        self.lr = LogisticRegression(C=args.C, max_iter=5000).fit(Xlr, ylr)
        self.keep = keep; bank, by = [], []
        for c in keep:
            Xc = Xlr[ylr == c]
            if args.kmeans and len(Xc) > args.kmeans:
                Xc = l2(KMeans(args.kmeans, n_init=1, random_state=args.seed).fit(Xc).cluster_centers_)
            bank.append(Xc); by += [c] * len(Xc)
        self.bank, self.by = np.concatenate(bank).astype(np.float32), np.array(by)
        sk, sl = self.raw(Xlr); self.st = (sk.mean(), sk.std(), sl.mean(), sl.std())
    def raw(self, X):
        lg = self.lr.decision_function(X)
        if lg.ndim == 1: lg = np.stack([-lg, lg], 1)          # binary LR (LOBO)
        pred = np.array(self.keep)[lg.argmax(1)]
        sims = X @ self.bank.T
        sims = np.where(self.by[None, :] == pred[:, None], sims, -2.0)
        sk = np.sort(sims, 1)[:, -args.k:].mean(1)
        return sk, lg.max(1)
    def __call__(self, X):
        sk, sl = self.raw(X); mk, sdk, ml, sdl = self.st
        lg = self.lr.decision_function(X)
        pred = np.array(self.keep)[(np.stack([-lg, lg], 1) if lg.ndim == 1 else lg).argmax(1)]
        return (sk - mk) / sdk + (sl - ml) / sdl, pred

folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(R, y))
oof_s, oof_ok = np.zeros(len(y)), np.zeros(len(y), bool)
zap = {"unk": [], "kn": []}; lobo = []
for tr, va in folds:
    H = Head(tr, [0, 1, 2]); s, pr = H(R[va]); oof_s[va] = s; oof_ok[va] = pr == y[va]
    if "zap_unk" in extra:
        zap["unk"].append(H(extra["zap_unk"])[0])
        zs, zp = H(extra["zap_known"]); zap["kn"].append((zs, zp == np.array([CLS.index(v) for v in extra["zap_known_lab"]])))
    for h in range(3):
        keep = [c for c in range(3) if c != h]; Hh = Head(tr, keep)
        kv = va[y[va] != h]; lobo.append((Hh(R[kv])[0], Hh(R[y == h])[0]))

thr = float(np.quantile(oof_s, args.q))
report = {"closed_set_acc_cv": float(oof_ok.mean()), "q": args.q, "thr": thr, "sweep": []}
print(f"CV closed-set accuracy {oof_ok.mean():.3f}; threshold (q={args.q}) = {thr:.3f}")
print(" q     thr    | CV known acc | LOBO real rej | Zappos test-brand rej | Zappos known acc")
for q in [0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15]:
    t = float(np.quantile(oof_s, q))
    acc = float(np.mean(oof_ok & (oof_s >= t)))
    lr_ = float(np.mean([np.mean(sh < np.quantile(sk, q)) for sk, sh in lobo]))   # per-run quantile, as in deployment
    zu = float(np.mean(np.concatenate(zap["unk"]) < t)) if zap["unk"] else float("nan")
    zk = float(np.mean(np.concatenate([o & (s >= t) for s, o in zap["kn"]]))) if zap["kn"] else float("nan")
    report["sweep"].append(dict(q=q, thr=t, cv_known_acc=acc, lobo_real_rej=lr_, zappos_unk_rej=zu, zappos_known_acc=zk))
    print(f" {q:.2f} {t:7.3f} |    {acc:.3f}     |    {lr_:.3f}      |        {zu:.3f}          |     {zk:.3f}" + ("   <- chosen" if abs(q - args.q) < 1e-9 else ""))

np.savez(args.out.replace(".npz", "_scores.npz"), thr=thr, oof_s=oof_s, oof_ok=oof_ok, y=y,
         lobo_known=np.concatenate([a for a, _ in lobo]), lobo_unk=np.concatenate([b for _, b in lobo]),
         zap_unk=np.concatenate(zap["unk"]) if zap["unk"] else np.zeros(0),
         zap_known=np.concatenate([a for a, _ in zap["kn"]]) if zap["kn"] else np.zeros(0))

# ---------------- final head on all real photos ----------------
H = Head(np.arange(len(y)), [0, 1, 2])
bank = H.bank; scale = np.abs(bank).max(1, keepdims=True) / 127.0
bank_q = np.round(bank / scale).clip(-127, 127).astype(np.int8)
np.savez(args.out, W=H.lr.coef_.astype(np.float32), b=H.lr.intercept_.astype(np.float32), bank_q=bank_q,
         bank_s=scale[:, 0].astype(np.float32), bank_y=H.by.astype(np.int64), k=args.k,
         stats=np.array(H.st, np.float32), thr=np.float32(thr))
# reference decisions of the final head on the training photos (for the export check)
s_all, p_all = H(R); dec = np.where(s_all < thr, 3, p_all)
np.save(args.out.replace(".npz", "_ref_decisions.npy"), dec)
report.update(bank_size=int(len(bank)), train_rejected=float(np.mean(dec == 3)))
json.dump(report, open(args.out.replace(".npz", "_report.json"), "w"), indent=1)
print(f"saved {args.out}: bank {bank.shape}, train-set rejected {np.mean(dec == 3):.3f}")
