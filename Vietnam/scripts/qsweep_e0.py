"""Sweep weight-quant policies for reparameterized MobileCLIP2-S0: est. size vs zero-shot OSR fidelity."""
import sys, csv, numpy as np, torch, timm, torch.nn.functional as F, copy
from PIL import Image
from timm.utils.model import reparameterize_model
from sklearn.metrics import roc_auc_score
dev = "cuda"
d = np.load("feats_e0_256.npz"); TB = torch.tensor(d["text_brand"]).float().to(dev); TE = torch.tensor(d["text_extra"]).float().to(dev)
K, N = TB[:3], torch.cat([TB[3:], TE])
tr = list(csv.DictReader(open("data/train-output.csv"))); y = d["real_y"]
zi = list(csv.DictReader(open("ext/zappos_index.csv")))
zk = [r["path"] for r in zi if r["role"] == "val_known" and r["split"] == "val"][::3]; zu = [r["path"] for r in zi if r["role"] == "val_unk"][::3]
zky = np.array([["adidas","converse","nike"].index(r["label"]) for r in zi if r["role"] == "val_known" and r["split"] == "val"][::3])
def load(ps): return torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)).copy()).permute(2, 0, 1) for p in ps])
XR, XK, XU = load(["data/train-input/" + r["filename"] for r in tr]), load(zk), load(zu)
base = reparameterize_model(timm.create_model("fastvit_mci0.apple_mclip2_dfndr2b", pretrained=True).eval())
sd0 = {k: v.clone() for k, v in base.state_dict().items()}
big = sorted([(v.numel(), k) for k, v in sd0.items() if v.dtype.is_floating_point and v.dim() >= 2], reverse=True)
print("total params %.2fM, >=2D %.2fM; top:" % (sum(v.numel() for v in sd0.values()) / 1e6, sum(n for n, _ in big) / 1e6), [(k, n) for n, k in big[:8]])
def q8(v):
    s = v.abs().flatten(1).amax(1).clamp_min(1e-12) / 127.0; s = s.view((-1,) + (1,) * (v.dim() - 1)); return torch.round(v / s).clamp(-127, 127) * s
def q4(v, G):
    w2 = v.flatten(1); n = w2.shape[1]; ng = (n + G - 1) // G; w3 = F.pad(w2, (0, ng * G - n)).view(w2.shape[0], ng, G)
    s = (w3.abs().amax(2, keepdim=True).clamp_min(1e-12) / 7.0).half().float(); return (torch.round(w3 / s).clamp(-7, 7) * s).reshape(w2.shape[0], -1)[:, :n].reshape(v.shape)
@torch.no_grad()
def emb(m, X):
    out = []
    for i in range(0, len(X), 64):
        x = X[i:i+64].to(dev).float() / 255
        f = F.normalize(m(x), dim=-1) + F.normalize(m(torch.flip(x, [3])), dim=-1); out.append(F.normalize(f, dim=-1))
    return torch.cat(out)
def score(f): k = f @ K.t(); return (k.max(1).values - (f @ N.t()).max(1).values).cpu().numpy(), k.argmax(1).cpu().numpy()
ref = None
def evaluate(name, sd, nbytes):
    global ref
    m = copy.deepcopy(base); m.load_state_dict(sd); m = m.to(dev).eval()
    fr, fk, fu = emb(m, XR), emb(m, XK), emb(m, XU)
    sr, pr = score(fr); sk, pk = score(fk); su, _ = score(fu)
    if ref is None: ref = fr
    cos = (fr * ref).sum(1).mean().item()
    t = np.quantile(sr, 0.05)
    print(f"{name:28s} ~{nbytes/1e6:5.2f}MB cos2fp={cos:.4f} AUC zk-zu={roc_auc_score(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[sk,su]):.4f} "
          f"| t(q05)={t:+.4f} knownacc real={np.mean((pr==y)&(sr>=t)):.3f} zk={np.mean((pk==zky)&(sk>=t)):.3f} rej zu={np.mean(su<t):.3f} | fixed t=-0.026: real={np.mean((pr==y)&(sr>=-0.026)):.3f} rej={np.mean(su<-0.026):.3f}", flush=True)
evaluate("fp32", sd0, 4 * sum(v.numel() for v in sd0.values()))
for pol in [("int8all", None, None), ("int4>=200k G32", 200000, 32), ("int4>=200k G16", 200000, 16), ("int4>=400k G16", 400000, 16), ("int4>=500k G16", 500000, 16), ("int4>=1M G16", 1000000, 16), ("int4>=400k G8", 400000, 8)]:
    name, thr, G = pol; sd = {}; nb = 0
    for k, v in sd0.items():
        if v.dtype.is_floating_point and v.dim() >= 2:
            if thr is not None and v.numel() >= thr: sd[k] = q4(v, G); nb += v.numel() / 2 + 2 * v.numel() / G
            else: sd[k] = q8(v); nb += v.numel() + 4 * v.shape[0]
        else: sd[k] = v; nb += 4 * v.numel()
    evaluate(name, sd, nb)
