"""Frozen-feature screening of small backbones for open-set sneaker recognition."""
import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v,"2")
import sys, csv, time, numpy as np, torch, timm, torch.nn.functional as F
torch.set_num_threads(2)
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
torch.backends.cudnn.benchmark = True
dev = "cuda"
CLS = ["adidas", "converse", "nike"]
tr = list(csv.DictReader(open("data/train-output.csv")))
tr_paths = ["data/train-input/" + r["filename"] for r in tr]
tr_y = np.array([CLS.index(r["label"]) for r in tr])
zi = list(csv.DictReader(open("ext/zappos_index.csv")))
zk = [r for r in zi if r["role"] == "val_known" and r["split"] == "val"]
zu = [r for r in zi if r["role"] == "val_unk"]
zo = [r for r in zi if r["role"] == "oe" and r["split"] == "val"][::3]
zk_y = np.array([CLS.index(r["label"]) for r in zk])
zu_brand = np.array([r["brand"].split()[0].lower() for r in zu])
def load(paths, size):
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
        out.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
    return torch.stack(out)
@torch.no_grad()
def feats(m, X, mean, std):
    fs = []
    for i in range(0, len(X), 128):
        x = X[i:i+128].to(dev).float() / 255
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            f = m.forward_head(m.forward_features(x), pre_logits=True)
            f2 = m.forward_head(m.forward_features(torch.flip(x, [3])), pre_logits=True)
        fs.append(((f + f2) / 2).float().cpu())
    return torch.cat(fs).numpy()
def l2(a): return a / np.linalg.norm(a, axis=1, keepdims=True)
def maha_fit(F_, y):
    mus = np.stack([F_[y == c].mean(0) for c in range(3)])
    Z = F_ - mus[y]; cov = Z.T @ Z / len(Z) + 1e-3 * np.eye(F_.shape[1]) * np.trace(Z.T @ Z / len(Z)) / F_.shape[1]
    P = np.linalg.inv(cov); return mus, P
def maha_score(F_, mus, P):
    d = np.stack([np.einsum("ij,jk,ik->i", F_ - m, P, F_ - m) for m in mus], 1)
    return -d.min(1), d.argmin(1)
def evaluate(name):
    m = timm.create_model(name, pretrained=True, num_classes=0).to(dev).eval()
    cfg = m.pretrained_cfg; size = cfg["input_size"][-1]; size = min(size, 256)
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    nparam = sum(p.numel() for p in m.parameters()) / 1e6
    Ftr = l2(feats(m, load(tr_paths, size), mean, std))
    Fzk = l2(feats(m, load([r["path"] for r in zk], size), mean, std))
    Fzu = l2(feats(m, load([r["path"] for r in zu], size), mean, std))
    Fzo = l2(feats(m, load([r["path"] for r in zo], size), mean, std))
    # 5-fold CV known accuracy on train (LR)
    skf = StratifiedKFold(5, shuffle=True, random_state=0); acc = []
    for a, b in skf.split(Ftr, tr_y):
        lr = LogisticRegression(C=2.0, max_iter=3000).fit(Ftr[a], tr_y[a]); acc.append((lr.predict(Ftr[b]) == tr_y[b]).mean())
    lr = LogisticRegression(C=2.0, max_iter=3000).fit(Ftr, tr_y)
    res = {"model": name, "Mparam": round(nparam, 2), "cv_acc": np.mean(acc)}
    res["zk_acc"] = (lr.predict(Fzk) == zk_y).mean()
    # scores: higher = more known
    msp = lambda F_: lr.predict_proba(F_).max(1)
    knn = lambda F_: np.sort(F_ @ Ftr.T, 1)[:, -5:].mean(1)
    mus, P = maha_fit(Ftr, tr_y); mh = lambda F_: maha_score(F_, mus, P)[0]
    for sn, fn in [("msp", msp), ("knn", knn), ("maha", mh)]:
        sk, su = fn(Fzk), fn(Fzu)
        res[f"auc_{sn}"] = roc_auc_score(np.r_[np.ones(len(sk)), np.zeros(len(su))], np.r_[sk, su])
    # OE-assisted: LR with 4th class from OE-train brands
    zt = [r for r in zi if r["role"] == "oe" and r["split"] == "train"][::4]
    Fot = l2(feats(m, load([r["path"] for r in zt], size), mean, std))
    lr4 = LogisticRegression(C=2.0, max_iter=3000).fit(np.r_[Ftr, Fot], np.r_[tr_y, np.full(len(Fot), 3)])
    p = lr4.predict_proba(Fzk); pu = lr4.predict_proba(Fzu)
    res["auc_oe4"] = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(pu))], np.r_[-p[:, 3], -pu[:, 3]])
    res["oe4_zk_acc"] = (p.argmax(1) == zk_y).mean(); res["oe4_zu_rej"] = (pu.argmax(1) == 3).mean()
    return res
names = sys.argv[1:]
for n in names:
    t = time.time()
    try:
        r = evaluate(n); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, f"{time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print(n, "ERR", repr(e)[:200], flush=True)
