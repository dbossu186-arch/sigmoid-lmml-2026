"""Distil the MobileCLIP2-S0 zero-shot open-set teacher into a pruned torchvision convnext_tiny (3 outputs).
Trains ONLY on the known-brand train images (strong aug + patch-shuffle/cutmix synthetic views); the teacher decides,
per augmented view, how 'known' it is: target = (1-pu)*softmax(ls*cos_known) + pu*uniform, pu = sigmoid((thr - margin)/T).
Zappos Puma/Reebok/Vans/Asics used for VALIDATION ONLY.
Usage: python distill_cnx.py --out runs/x --holdout 0.15 --keep2 4 --epochs 60"""
from torchvision.transforms import v2
import argparse, os, csv, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm, torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
p = argparse.ArgumentParser()
p.add_argument("--out", required=True); p.add_argument("--holdout", type=float, default=0.15); p.add_argument("--keep2", type=int, default=4)
p.add_argument("--keep1", type=int, default=3); p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--thr", type=float, default=-0.0262); p.add_argument("--T", type=float, default=0.006)
p.add_argument("--synth", type=float, default=0.3); p.add_argument("--feat", type=float, default=0.0); p.add_argument("--erase", type=float, default=0.0); p.add_argument("--erase_max", type=float, default=0.2); p.add_argument("--blur", type=float, default=0.0); p.add_argument("--tflip", type=int, default=1); p.add_argument("--ema", type=float, default=0.0); p.add_argument("--logo", type=float, default=0.0, help="prob. of hiding/swapping the brand-mark region"); p.add_argument("--gen", type=int, default=0, help="generic images added per batch (Caltech101, teacher-labelled)"); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(f"{a.out}/args.json", "w"))
torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"
CLS = ["adidas", "converse", "nike"]
tr = list(csv.DictReader(open("data/train-output.csv"))); y_all = np.array([CLS.index(r["label"]) for r in tr])
rng = np.random.RandomState(0); va_idx = []
if a.holdout > 0:
    for c in range(3):
        ii = np.where(y_all == c)[0]; rng.shuffle(ii); va_idx += list(ii[: int(round(len(ii) * a.holdout))])
va_idx = np.array(sorted(va_idx), dtype=int); tr_idx = np.array([i for i in range(len(tr)) if i not in set(va_idx)])
def load(ps, size=256): return torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)).copy()).permute(2, 0, 1) for p in ps])
X = load(["data/train-input/" + r["filename"] for r in tr])
zi = list(csv.DictReader(open("ext/zappos_index.csv")))
zkr = [r for r in zi if r["role"] == "val_known" and r["split"] == "val"][::4]; zur = [r for r in zi if r["role"] == "val_unk"][::5]
XK, XU = load([r["path"] for r in zkr]), load([r["path"] for r in zur]); zky = np.array([CLS.index(r["label"]) for r in zkr])
LM = torch.tensor(np.load("logo_map.npy")) if a.logo > 0 else None
def hide_logo(xu8, idx):
    """Hide the top brand-evidence windows (from logo_map.npy): grey, heavy blur, or another brand's mark region."""
    out = xu8.clone()
    for n, i in enumerate(idx):
        if np.random.rand() >= a.logo: continue
        order = torch.argsort(LM[i].flatten(), descending=True)[: np.random.randint(1, 4)]
        mode = np.random.rand()
        if mode > 0.66:  # swap in a mark region from an image of another brand
            j = np.random.choice(np.where(y_all != y_all[i])[0]); oj = int(torch.argmax(LM[j]))
            src = X[j][:, (oj // 7) * 32:(oj // 7) * 32 + 64, (oj % 7) * 32:(oj % 7) * 32 + 64]
        for o in order.tolist():
            gy, gx = (o // 7) * 32, (o % 7) * 32
            if mode < 0.33: out[n, :, gy:gy+64, gx:gx+64] = 128
            elif mode < 0.66: out[n, :, gy:gy+64, gx:gx+64] = blurrer(out[n, :, gy:gy+64, gx:gx+64])
            else: out[n, :, gy:gy+64, gx:gx+64] = src; break
    return out
XG = None
if a.gen:  # generic non-shoe images as extra unlabeled inputs
    import glob
    gp = sorted(glob.glob("gen/caltech101/101_ObjectCategories/*/*.jpg")); np.random.RandomState(1).shuffle(gp)
    XG = load(gp[:4000]); print("generic images", len(XG), flush=True)
# ---- teacher
d = np.load("feats_e0_256.npz"); TB = torch.tensor(d["text_brand"]).float().to(dev); TE = torch.tensor(d["text_extra"]).float().to(dev)
TK, TN, LS = TB[:3], torch.cat([TB[3:], TE]), float(d["logit_scale"])
teacher = timm.create_model("fastvit_mci0.apple_mclip2_dfndr2b", pretrained=True).to(dev).eval().half()
@torch.no_grad()
def teach(x01):  # x01 float [0,1] on dev, 256
    f = F.normalize(teacher(x01.half()).float(), dim=-1)
    if a.tflip: f = F.normalize(f + F.normalize(teacher(torch.flip(x01, [3]).half()).float(), dim=-1), dim=-1)
    k = f @ TK.t(); m = k.max(1).values - (f @ TN.t()).max(1).values
    pu = torch.sigmoid((a.thr - m) / a.T)
    return (1 - pu)[:, None] * torch.softmax(LS * k, 1) + pu[:, None] / 3.0, pu, f
# ---- student
S = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
S.classifier[2] = nn.Linear(768, 3)
keep = {1: 3, 3: a.keep1, 5: a.keep2, 7: 0}
for si, k in keep.items():
    for j in range(k, len(S.features[si])): S.features[si][j] = nn.Identity()
S = S.to(dev).to(memory_format=torch.channels_last)
proj = nn.Linear(768, 512).to(dev)  # training-only: maps pooled student features onto the teacher embedding
eraser = v2.RandomErasing(p=1.0, scale=(0.02, a.erase_max), ratio=(0.3, 3.3), value=0.5)
blurrer = v2.GaussianBlur(kernel_size=21, sigma=(2.0, 8.0))
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
aug = v2.Compose([v2.RandomResizedCrop(256, scale=(0.3, 1.0), ratio=(0.7, 1.4), antialias=True), v2.RandomHorizontalFlip(),
                  v2.RandomApply([v2.ColorJitter(0.35, 0.35, 0.35, 0.05)], p=0.8), v2.RandomGrayscale(0.1),
                  v2.RandomApply([v2.RandomRotation(15)], p=0.3)])
def synth(xb):  # xb float [0,1] B,3,H,W : make some 'unknown-ish' views (teacher will label them)
    B = len(xb); out = xb.clone()
    for i in range(B):
        r = np.random.rand()
        if r < a.synth / 2:  # patch shuffle
            g = np.random.choice([2, 4]); h = 256 // g
            P = out[i].unfold(1, h, h).unfold(2, h, h).reshape(3, g * g, h, h)[:, torch.randperm(g * g)]
            out[i] = P.reshape(3, g, g, h, h).permute(0, 1, 3, 2, 4).reshape(3, 256, 256)
        elif r < a.synth:  # cutmix with another image
            j = np.random.randint(B); s = np.random.randint(96, 192); y0, x0 = np.random.randint(0, 256 - s, 2)
            out[i, :, y0:y0+s, x0:x0+s] = xb[j, :, y0:y0+s, x0:x0+s]
        if np.random.rand() < a.blur: out[i] = blurrer(out[i])   # logo no longer readable
        if np.random.rand() < a.erase:
            for _ in range(np.random.randint(1, 4)): out[i] = eraser(out[i])
    return out
params = [{"params": [p_ for n, p_ in S.named_parameters() if not n.startswith("classifier")], "lr": a.lr},
          {"params": list(S.classifier.parameters()) + list(proj.parameters()), "lr": a.lr * 5}]
opt = torch.optim.AdamW(params, weight_decay=0.05)
steps = a.epochs * (len(tr_idx) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 5], total_steps=steps, pct_start=0.1)
scaler = torch.amp.GradScaler()
import copy
E = copy.deepcopy(S).eval() if a.ema > 0 else None  # weight EMA, used for eval and saving
@torch.no_grad()
def predict(Xu8):
    M = E if E is not None else S
    S.eval(); out = []
    for i in range(0, len(Xu8), 64):
        x = (Xu8[i:i+64].to(dev).float() / 255 - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.float16): out.append(torch.softmax(M(x.contiguous(memory_format=torch.channels_last)).float(), 1))
    S.train(); return torch.cat(out).cpu().numpy()
def report(tag):
    res = {}
    pk, pu_ = predict(XK), predict(XU); mk, mu = pk.max(1), pu_.max(1)
    auc = roc_auc_score(np.r_[np.ones(len(mk)), np.zeros(len(mu))], np.r_[mk, mu])
    line = f"[{tag}] zk-acc={np.mean(pk.argmax(1)==zky):.3f} AUC(zk,zu)={auc:.3f}"
    if len(va_idx):
        pv = predict(X[va_idx]); mv = pv.max(1); yv = y_all[va_idx]
        line += f" va-acc={np.mean(pv.argmax(1)==yv):.3f} AUC(va,zu)={roc_auc_score(np.r_[np.ones(len(mv)),np.zeros(len(mu))],np.r_[mv,mu]):.3f}"
        for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
            line += f"\n   tau={tau}: va knownacc={np.mean((pv.argmax(1)==yv)&(mv>=tau)):.3f} zk knownacc={np.mean((pk.argmax(1)==zky)&(mk>=tau)):.3f} zu rej={np.mean(mu<tau):.3f}"
        res["va"] = pv; res["va_y"] = yv
    print(line, flush=True); res.update(zk=pk, zk_y=zky, zu=pu_); return res
t0 = time.time(); step = 0; S.train()
for ep in range(a.epochs):
    perm = np.random.permutation(tr_idx); tot = 0; pus = 0
    for b in range(len(perm) // a.bs):
        idx = perm[b * a.bs:(b + 1) * a.bs]
        xb = X[idx].to(dev)
        if LM is not None: xb = hide_logo(xb, idx)
        xb = torch.stack([aug(x) for x in xb]).float() / 255
        xb = synth(xb).clamp(0, 1)
        if XG is not None:
            xg = XG[np.random.randint(len(XG), size=a.gen)].to(dev)
            xb = torch.cat([xb, torch.stack([aug(x) for x in xg]).float() / 255])
        tgt, pu, temb = teach(xb)
        with torch.autocast("cuda", dtype=torch.float16):
            h = S.classifier[1](S.classifier[0](S.avgpool(S.features(((xb - MEAN) / STD).contiguous(memory_format=torch.channels_last)))))
            lo = S.classifier[2](h).float(); fe = proj(h).float()
        loss = F.kl_div(F.log_softmax(lo, 1), tgt, reduction="batchmean") + a.feat * (1 - F.cosine_similarity(fe, temb, dim=1)).mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if E is not None:
            with torch.no_grad():
                for pe, ps in zip(E.state_dict().values(), S.state_dict().values()):
                    if pe.dtype.is_floating_point: pe.mul_(a.ema).add_(ps.detach(), alpha=1 - a.ema)
                    else: pe.copy_(ps)
        tot += loss.item(); pus += pu.mean().item(); step += 1
    print(f"ep {ep} loss {tot/(b+1):.4f} teacher-pu {pus/(b+1):.3f} {time.time()-t0:.0f}s", flush=True)
    if (ep + 1) % 15 == 0 or ep == a.epochs - 1: res = report(f"ep{ep}")
torch.save({"state_dict": (E if E is not None else S).state_dict(), "keep": keep}, f"{a.out}/student.pth")
np.savez(f"{a.out}/val_probs.npz", **res); print("done", time.time() - t0)
