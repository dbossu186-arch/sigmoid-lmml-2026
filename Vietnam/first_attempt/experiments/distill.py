"""Distil a CLIP image encoder (teacher, frozen) into a small student CNN that fits in 10 MB.

Unlabelled distillation data (no brand labels are used here):
  * the provided real photos (fold-0 validation photos excluded so the proxy stays honest),
  * UT-Zappos50K shoes of brands that are NOT test brands (PUMA/Reebok/Vans/ASICS/Onitsuka never used),
  * with --holdout B, every image of known brand B (real + Zappos) is excluded too (LOBO proxy).
Student and teacher see the same augmented crop; loss = 1 - cos(student_proj(x), teacher(x)).
At the end: frozen-feature LOBO metrics on REAL photos (same protocol as lobo_frozen.py).
Usage: python distill.py --out runs/dist_b0 [--holdout nike] [--teacher vit_base_patch16_clip_224.openai]
"""
import argparse, csv, json, os, random, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--teacher", default="vit_base_patch16_clip_224.openai")
p.add_argument("--student", default="efficientnet_b0.ra_in1k")
p.add_argument("--holdout", default="")
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--size", type=int, default=256)
p.add_argument("--real_rep", type=int, default=6, help="how often each real photo is repeated per epoch")
p.add_argument("--zap_cap", type=int, default=16000, help="max Zappos images per epoch")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--workers", type=int, default=2)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True); json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.set_num_threads(2); dev = "cuda"
CLS = ["adidas", "converse", "nike"]
TEST_UNK = ["puma", "reebok", "vans", "asics", "onitsuka"]

rows = list(csv.DictReader(open("data/train-output.csv")))
y = np.array([CLS.index(r["label"]) for r in rows])
va = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))[0][1]
isva = np.zeros(len(y), bool); isva[va] = True
real_paths = ["data/train-input/" + r["filename"] for r in rows]
real_dist = [real_paths[i] for i in range(len(rows)) if not isva[i] and rows[i]["label"] != args.holdout]
zroot = "ext/ut-zap50k-images-square"
zap = []
for dp, _, fs in os.walk(zroot):
    brand = dp.split(os.sep)[-1].lower() if dp.count(os.sep) >= 3 else ""
    if not brand or any(k in brand for k in TEST_UNK): continue
    if args.holdout and args.holdout in brand: continue
    zap += [os.path.join(dp, f) for f in fs if f.endswith(".jpg")]
zap.sort(); random.Random(0).shuffle(zap)
print(f"distil set: {len(real_dist)} real (x{args.real_rep}) + {len(zap)} Zappos (cap {args.zap_cap}/epoch)", flush=True)

aug = T.Compose([T.RandomResizedCrop(args.size, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                 T.RandomHorizontalFlip(), T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.6), T.ToTensor()])
class DS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return aug(Image.open(self.paths[i]).convert("RGB"))

teacher = timm.create_model(args.teacher, pretrained=True, num_classes=0).to(dev).eval().half()
for q in teacher.parameters(): q.requires_grad_(False)
tcfg = teacher.pretrained_cfg; tsize = tcfg["input_size"][-1]
tmean = torch.tensor(tcfg["mean"], device=dev).view(1, 3, 1, 1); tstd = torch.tensor(tcfg["std"], device=dev).view(1, 3, 1, 1)
smean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); sstd = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
tdim = teacher.num_features

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = timm.create_model(args.student, pretrained=True, num_classes=0)
        self.proj = nn.Linear(self.body.num_features, tdim)
    def forward(self, x): return self.proj(self.body(x))
student = Student().to(dev)
student.body.set_grad_checkpointing(True)

@torch.no_grad()
def t_embed(x01):
    xt = F.interpolate(x01, size=(tsize, tsize), mode="bicubic", align_corners=False, antialias=True)
    return teacher(((xt - tmean) / tstd).half()).float()

steps_per_epoch = (len(real_dist) * args.real_rep + min(len(zap), args.zap_cap)) // args.bs
opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.02)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
scaler = torch.amp.GradScaler(); t0 = time.time()
for ep in range(args.epochs):
    zs = zap[(ep * args.zap_cap) % max(len(zap), 1):][:args.zap_cap]
    if len(zs) < min(len(zap), args.zap_cap): zs += zap[:args.zap_cap - len(zs)]
    paths = real_dist * args.real_rep + zs; random.shuffle(paths)
    dl = DataLoader(DS(paths), batch_size=args.bs, shuffle=False, num_workers=args.workers, drop_last=True,
                    multiprocessing_context="fork")
    tl, n = 0, 0
    for x in dl:
        x = x.to(dev, non_blocking=True)
        tgt = F.normalize(t_embed(x), dim=1)
        with torch.autocast("cuda", dtype=torch.float16):
            s = student((x - smean) / sstd)
        loss = (1 - F.cosine_similarity(s.float(), tgt, dim=1)).mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched.last_epoch < args.epochs * steps_per_epoch - 1: sched.step()
        tl += loss.item(); n += 1
    print(f"ep {ep} cos-loss {tl/max(n,1):.4f} {time.time()-t0:.0f}s", flush=True)
torch.save({"state_dict": student.state_dict(), "student": args.student, "teacher": args.teacher, "tdim": tdim, "size": args.size},
           f"{args.out}/student.pth")

# ---- frozen LOBO evaluation on real photos (student embedding vs teacher embedding) ----
student.eval()
@torch.no_grad()
def embed_all(fn):
    out = []
    for i in range(0, len(real_paths), 16):
        x = torch.stack([T.functional.to_tensor(Image.open(q).convert("RGB").resize((args.size, args.size), Image.BICUBIC))
                         for q in real_paths[i:i + 16]]).to(dev)
        out.append(((fn(x) + fn(torch.flip(x, [3]))) / 2).float().cpu())
    e = torch.cat(out).numpy(); return e / np.linalg.norm(e, axis=1, keepdims=True)
def s_fn(x):
    with torch.autocast("cuda", dtype=torch.float16): return student((x - smean) / sstd)
res = {}
for nm, fn in [("student", s_fn), ("teacher", t_embed)]:
    Fe = embed_all(fn); np.save(f"{args.out}/real_emb_{nm}.npy", Fe); line = [nm]
    for ho in range(3):
        tr = (~isva) & (y != ho); kv = isva & (y != ho); un = y == ho
        knn = lambda Q: np.sort(Q @ Fe[tr].T, 1)[:, -5:].mean(1)
        a = roc_auc_score(np.r_[np.ones(kv.sum()), np.zeros(un.sum())], np.r_[knn(Fe[kv]), knn(Fe[un])])
        line.append(f"ho={CLS[ho]} knn={a:.3f}")
    lr = LogisticRegression(C=2, max_iter=3000).fit(Fe[~isva], y[~isva]); line.append(f"3cls_acc={np.mean(lr.predict(Fe[isva]) == y[isva]):.3f}")
    print(" | ".join(line), flush=True)
print("done", time.time() - t0)
