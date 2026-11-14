"""LOBO upper-bound check with frozen encoders on REAL photos (Mac, MPS). kNN / LR-MSP AUROC for held-out brand."""
import sys, csv, numpy as np, torch, timm
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
torch.set_num_threads(2); dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = list(csv.DictReader(open("data/train-output.csv"))); CLS = ["adidas", "converse", "nike"]
y = np.array([CLS.index(r["label"]) for r in rows])
va = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))[0][1]; isva = np.zeros(len(y), bool); isva[va] = True
for name in sys.argv[1:]:
    kw = {"img_size": 224} if "dinov2" in name else {}
    m = timm.create_model(name, pretrained=True, num_classes=0, **kw).to(dev).eval()
    cfg = m.pretrained_cfg; s = (224 if "dinov2" in name else cfg["input_size"][-1])
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
    F = []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            x = torch.stack([torch.from_numpy(np.asarray(Image.open("data/train-input/" + r["filename"]).convert("RGB").resize((s, s), Image.BICUBIC)).copy()).permute(2, 0, 1) for r in rows[i:i + 16]]).float().to(dev) / 255
            x = (x - mean) / std; F.append(((m(x) + m(torch.flip(x, [3]))) / 2).cpu())
    F = torch.cat(F).numpy(); F /= np.linalg.norm(F, axis=1, keepdims=True)
    np.save(f"runs/feats/{name}.npy", F)
    out = [f"{name} ({sum(p.numel() for p in m.parameters())/1e6:.2f}M)"]
    for ho in range(3):
        tr = (~isva) & (y != ho); kv = isva & (y != ho); un = y == ho
        knn = lambda Q: np.sort(Q @ F[tr].T, 1)[:, -5:].mean(1)
        a_knn = roc_auc_score(np.r_[np.ones(kv.sum()), np.zeros(un.sum())], np.r_[knn(F[kv]), knn(F[un])])
        lr = LogisticRegression(C=2, max_iter=3000).fit(F[tr], y[tr])
        a_msp = roc_auc_score(np.r_[np.ones(kv.sum()), np.zeros(un.sum())], np.r_[lr.predict_proba(F[kv]).max(1), lr.predict_proba(F[un]).max(1)])
        out.append(f"ho={CLS[ho]} knn={a_knn:.3f} msp={a_msp:.3f}")
    lr = LogisticRegression(C=2, max_iter=3000).fit(F[~isva], y[~isva]); out.append(f"3cls_acc={np.mean(lr.predict(F[isva]) == y[isva]):.3f}")
    print(" | ".join(out), flush=True)
