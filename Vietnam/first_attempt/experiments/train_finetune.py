"""Open-set sneaker brand recogniser (adidas / converse / nike / unknown).

Fine-tunes a small ImageNet-pretrained timm backbone on
  * the 735 provided real photos (known brands),
  * optional UT-Zappos50K Nike/adidas/Converse catalog images (extra knowns, product-level split),
  * optional outlier exposure (OE) from UT-Zappos50K sneakers of OTHER brands
    (never PUMA / Reebok / Vans / ASICS / Onitsuka, which are kept for validation only),
with optional compositing of Zappos shoes onto blurred real-photo backgrounds so that
"catalog look" cannot become a shortcut for "unknown".

Every run dumps logits for all validation sets to <out>/val_logits.npz for threshold analysis.

Usage examples (run on the GPU box):
  python train.py --out runs/cv0 --fold 0                  # 5-fold CV, fold 0
  python train.py --out runs/lobo_nike --holdout nike      # leave-one-brand-out proxy
  python train.py --out runs/final --fold -1               # final model on all real data
"""
import argparse, csv, json, os, random, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "2")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold

CLS = ["adidas", "converse", "nike"]
p = argparse.ArgumentParser()
p.add_argument("--model", default="efficientnet_b0.ra_in1k")
p.add_argument("--out", required=True)
p.add_argument("--fold", type=int, default=0, help="CV fold 0-4 of real data held out; -1 = use all")
p.add_argument("--holdout", default="", help="known brand treated as unknown (LOBO proxy)")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--steps", type=int, default=150, help="optimizer steps per epoch")
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=4e-4)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--size", type=int, default=256)
p.add_argument("--zk", type=float, default=0.25, help="sampling share of Zappos knowns (0 = off)")
p.add_argument("--oe", type=float, default=0.35, help="sampling share of OE outliers (0 = off)")
p.add_argument("--oe_mode", default="class", choices=["class", "uniform"], help="4th class vs. uniform-target OE")
p.add_argument("--oe_w", type=float, default=0.5, help="weight of uniform OE loss")
p.add_argument("--composite", type=float, default=0.5, help="prob. of pasting a Zappos shoe on a real background")
p.add_argument("--ls", type=float, default=0.1)
p.add_argument("--pair", type=float, default=0.0, help="prob. that a composite shows a pair of shoes")
p.add_argument("--persp", type=float, default=0.0, help="prob. of random perspective warp")
p.add_argument("--degrade", type=float, default=0.0, help="prob. of random down/up-sampling (96-180 px) for ALL images")
p.add_argument("--composite_real", type=float, default=0.0,
               help="prob. of compositing white-background REAL knowns onto blurred backgrounds (same artifact as OE)")
p.add_argument("--cam_oe", type=float, default=0.0,
               help="share of real known images per batch turned into 'evidence-erased' outliers (CAM peak blurred out)")
p.add_argument("--cam_area", type=float, default=0.15, help="fraction of image area erased around the CAM peak")
p.add_argument("--cam_warm", type=int, default=2, help="epochs before CAM outliers are used")
p.add_argument("--drop_path", type=float, default=0.1)
p.add_argument("--grad_ckpt", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--workers", type=int, default=2)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(args.seed)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- data ----------------
real = [("data/train-input/" + r["filename"], r["label"]) for r in csv.DictReader(open("data/train-output.csv"))]
real_y = np.array([CLS.index(l) for _, l in real])
if args.fold >= 0:
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    tr_idx, va_idx = list(skf.split(np.zeros(len(real)), real_y))[args.fold]
else:
    tr_idx, va_idx = np.arange(len(real)), np.array([], int)
known = [c for c in CLS if c != args.holdout]
NK = len(known)  # number of known classes in this run
UNK = NK         # index of the explicit unknown class (oe_mode=class)
real_tr = [(real[i][0], known.index(real[i][1])) for i in tr_idx if real[i][1] in known]
real_va = [(real[i][0], real[i][1]) for i in va_idx]
real_ho = [(pth, l) for pth, l in real if l == args.holdout]  # LOBO unknown proxy (real photos)

zi = list(csv.DictReader(open("ext/zappos_index.csv"))) if os.path.exists("ext/zappos_index.csv") else []
zk_tr = [(r["path"], known.index(r["label"])) for r in zi if r["role"] == "val_known" and r["split"] == "train" and r["label"] in known]
oe_tr = [(r["path"], -1) for r in zi if r["role"] == "oe" and r["split"] == "train"]
zk_va = [(r["path"], r["label"]) for r in zi if r["role"] == "val_known" and r["split"] == "val"]
zu_va = [(r["path"], r["brand"]) for r in zi if r["role"] == "val_unk"]
oe_va = [(r["path"], r["brand"]) for r in zi if r["role"] == "oe" and r["split"] == "val"]
if args.holdout:  # Zappos images of the held-out brand also become unknown-proxy, never training data
    zk_tr = [(r["path"], known.index(r["label"])) for r in zi if r["role"] == "val_known" and r["label"] in known and r["split"] == "train"]

bg_pool = [pth for pth, _ in real_tr]
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
aug = T.Compose([
    T.RandomResizedCrop(args.size, scale=(0.35, 1.0), ratio=(0.75, 1.33), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(), T.RandomApply([T.RandomRotation(20)], p=0.3), T.RandomPerspective(0.3, p=args.persp),
    T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.8), T.RandomGrayscale(0.05),
    T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(p=0.2, scale=(0.02, 0.12))])
plain = T.Compose([T.Resize((args.size, args.size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean, std)])

def composite(im, rng):
    """Paste a white-background catalog shoe onto a blurred random real-photo background."""
    im = im.convert("RGB").resize((256, 256), Image.BICUBIC)
    a = np.asarray(im).astype(np.float32)
    border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3), a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    bgc = np.median(border, 0)
    mask = (np.abs(a - bgc).max(2) > 18).astype(np.uint8) * 255  # adaptive to the (near-)uniform background colour
    m = Image.fromarray(mask).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))
    bg = Image.open(rng.choice(bg_pool)).convert("RGB").resize((256, 256))
    bg = bg.rotate(rng.choice([0, 90, 180, 270])).filter(ImageFilter.GaussianBlur(rng.uniform(6, 14)))
    if rng.random() < args.pair:  # a pair of shoes, as in many real photos
        s = rng.uniform(0.5, 0.75); w = int(256 * s)
        fg, mm = im.resize((w, w)), m.resize((w, w))
        flip = rng.random() < 0.5
        fg2 = fg.transpose(Image.FLIP_LEFT_RIGHT) if flip else fg; m2 = mm.transpose(Image.FLIP_LEFT_RIGHT) if flip else mm
        x, y = rng.randint(0, 256 - w), rng.randint(0, 256 - w)
        x2 = min(max(x + rng.randint(-w // 3, w // 3), 0), 256 - w); y2 = min(max(y + rng.choice([-1, 1]) * rng.randint(w // 5, w // 2), 0), 256 - w)
        bg.paste(fg2, (x2, y2), m2); bg.paste(fg, (x, y), mm)
        return bg
    s = rng.uniform(0.6, 1.0); w = int(256 * s)
    fg, m = im.resize((w, w)), m.resize((w, w))
    x, y = rng.randint(0, 256 - w), rng.randint(0, 256 - w)
    bg.paste(fg, (x, y), m)
    return bg

class TrainDS(Dataset):
    def __init__(self, items, zappos):
        self.items, self.zappos = items, zappos
        self._wb = {}
    def white_bg(self, i, im):
        if i not in self._wb:
            a = np.asarray(im.resize((64, 64))).min(2)
            border = np.r_[a[0], a[-1], a[:, 0], a[:, -1]]
            self._wb[i] = (border > 230).mean() > 0.85
        return self._wb[i]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        pth, y = self.items[i]
        im = Image.open(pth).convert("RGB")
        if self.zappos[i] and random.random() < args.composite:
            im = composite(im, random)
        elif not self.zappos[i] and args.composite_real > 0 and self.white_bg(i, im) and random.random() < args.composite_real:
            im = composite(im, random)
        if args.degrade > 0 and random.random() < args.degrade:
            r = random.randint(96, 180)
            im = im.resize((r, r), random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])).resize((256, 256), Image.BICUBIC)
        return aug(im), y, int(not self.zappos[i])

items = real_tr + (zk_tr if args.zk > 0 else []) + (oe_tr if args.oe > 0 else [])
zflag = [False] * len(real_tr) + [True] * (len(items) - len(real_tr))
share_real = 1 - (args.zk if zk_tr else 0) - (args.oe if args.oe > 0 else 0)
w = [share_real / len(real_tr)] * len(real_tr)
if args.zk > 0: w += [args.zk / len(zk_tr)] * len(zk_tr)
if args.oe > 0: w += [args.oe / len(oe_tr)] * len(oe_tr)
sampler = WeightedRandomSampler(w, num_samples=args.steps * args.bs, replacement=True,
                                generator=torch.Generator().manual_seed(args.seed))
def wif(i): s = args.seed * 1000 + i + torch.initial_seed() % 1000; random.seed(s); np.random.seed(s % 2**32)
dl = DataLoader(TrainDS(items, zflag), batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                drop_last=True, worker_init_fn=wif, persistent_workers=args.workers > 0,
                multiprocessing_context="fork" if args.workers > 0 else None)

class EvalDS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return plain(Image.open(self.paths[i]).convert("RGB"))

# ---------------- model ----------------
n_out = NK + 1 if ((args.oe > 0 or args.cam_oe > 0) and args.oe_mode == "class") else NK
model = timm.create_model(args.model, pretrained=True, num_classes=n_out, drop_path_rate=args.drop_path).to(dev)
if args.grad_ckpt: model.set_grad_checkpointing(True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
total = args.epochs * args.steps
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos")
scaler = torch.amp.GradScaler()

def loss_fn(logits, y):
    kn = y >= 0
    loss = torch.zeros((), device=logits.device)
    if args.oe_mode == "class" or args.oe <= 0:
        yy = torch.where(kn, y, torch.full_like(y, UNK))
        return F.cross_entropy(logits, yy, label_smoothing=args.ls)
    if kn.any(): loss = loss + F.cross_entropy(logits[kn], y[kn], label_smoothing=args.ls)
    if (~kn).any(): loss = loss + args.oe_w * (-F.log_softmax(logits[~kn], 1).mean(1)).mean()
    return loss

@torch.no_grad()
def predict(paths, feats=False):
    if not paths: return (np.zeros((0, n_out), np.float32), np.zeros((0, model.num_features), np.float32)) if feats else np.zeros((0, n_out), np.float32)
    model.eval(); out = []; fo = []
    for x in DataLoader(EvalDS(paths), batch_size=32, num_workers=args.workers,
                        multiprocessing_context="fork" if args.workers > 0 else None):
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            f = (model.forward_head(model.forward_features(x), pre_logits=True) +
                 model.forward_head(model.forward_features(torch.flip(x, [3])), pre_logits=True)) / 2
            l = model.get_classifier()(f)  # linear head => same as averaging logits
        out.append(l.float().cpu()); fo.append(f.float().cpu())
    model.train()
    return (torch.cat(out).numpy(), torch.cat(fo).numpy()) if feats else torch.cat(out).numpy()

@torch.no_grad()
def cam_erase(x, y, control=False):
    """Blur out the most class-discriminative region (CAM of the true class) -> 'shoe without its tells'."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        fm = model.forward_features(x).float()                       # [B,C,h,w]
    W = model.get_classifier().weight.float()                         # [n_out,C] (linear head after GAP)
    if fm.shape[1] != W.shape[1]:                                     # e.g. MobileNetV3: conv_head between; fall back
        fm = model.conv_head(fm) if hasattr(model, "conv_head") else fm
        fm = model.act2(fm) if hasattr(model, "act2") else fm
    cam = torch.einsum("bchw,bc->bhw", fm, W[y]).unsqueeze(1)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    q = torch.quantile(cam.flatten(1), 1 - args.cam_area, dim=1).view(-1, 1, 1, 1)
    m = (cam >= q).float()
    if control:  # same-shaped blur patch moved away from the evidence peak -> label stays known
        h, w = m.shape[-2:]
        m = torch.roll(m, shifts=(int(h * (0.3 + 0.4 * random.random())), int(w * (0.3 + 0.4 * random.random()))), dims=(2, 3)) * (1 - m)
    m = F.max_pool2d(m, 9, 1, 4); m = F.avg_pool2d(m, 9, 1, 4)        # dilate + soften edges
    blur = F.avg_pool2d(F.avg_pool2d(x, 15, 1, 7, count_include_pad=False), 15, 1, 7, count_include_pad=False)
    return x * (1 - m) + blur * m

t0 = time.time(); step = 0
model.train()
for ep in range(args.epochs):
    tl = 0
    for x, y, isreal in dl:
        x, y, isreal = x.to(dev, non_blocking=True), y.to(dev), isreal.to(dev)
        if args.cam_oe > 0 and ep >= args.cam_warm:
            sel = ((isreal == 1) & (y >= 0)).nonzero().flatten()
            sel = sel[torch.rand(len(sel), device=dev) < args.cam_oe]
            if len(sel):
                xc = cam_erase(x[sel], y[sel])
                x = torch.cat([x, xc]); y = torch.cat([y, torch.full((len(sel),), -1, device=dev, dtype=y.dtype)])
                ctl = ((isreal == 1) & (y[:len(isreal)] >= 0)).nonzero().flatten()
                ctl = ctl[torch.randperm(len(ctl), device=dev)[:len(sel)]]
                x[ctl] = cam_erase(x[ctl], y[ctl], control=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            loss = loss_fn(model(x).float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item(); step += 1
    msg = f"ep {ep} loss {tl/args.steps:.3f} {time.time()-t0:.0f}s"
    if real_va and (ep % 5 == 4 or ep == args.epochs - 1):
        lv = predict([q for q, _ in real_va])
        yv = np.array([known.index(l) if l in known else -1 for _, l in real_va])
        msg += f" real_va_acc {(lv[:, :NK].argmax(1) == yv)[yv >= 0].mean():.3f}"
    print(msg, flush=True)

torch.save({"state_dict": model.state_dict(), "model": args.model, "known": known, "n_out": n_out, "size": args.size},
           f"{args.out}/model.pth")
res = {"known": np.array(known)}
for name, lst in [("real_va", real_va), ("real_ho", real_ho), ("zk_va", zk_va), ("zu_va", zu_va), ("oe_va", oe_va),
                  ("real_tr", [(q, known[y]) for q, y in real_tr])]:
    res[name], res[name + "_feat"] = predict([q for q, _ in lst], feats=True); res[name + "_lab"] = np.array([l for _, l in lst])
np.savez(f"{args.out}/val_logits.npz", **res)
print("done", time.time() - t0, flush=True)
