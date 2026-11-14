"""Extract frozen backbone features (flip-TTA) for the real photos (+ augmented views) and Zappos sets.
Usage: python extract_feats.py <timm_model> <out.npz> [n_aug_views]"""
import os, sys, csv, random
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(_v, "2")
import numpy as np, torch, timm
from PIL import Image
import torchvision.transforms as T
from timm.utils.model import reparameterize_model
torch.set_num_threads(2); dev = "cuda"; random.seed(0); torch.manual_seed(0)
name, out = sys.argv[1], sys.argv[2]; n_aug = int(sys.argv[3]) if len(sys.argv) > 3 else 4
if name.endswith(".pt"):   # exported TorchScript backbone (expects [0,1] input, MobileCLIP mean 0 / std 1)
    m = torch.jit.load(name, map_location=dev).eval(); cfgm, cfgs = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0); AMP = False
else:
    m = reparameterize_model(timm.create_model(name, pretrained=True, num_classes=0).eval()).to(dev)
    cfgm, cfgs = m.pretrained_cfg["mean"], m.pretrained_cfg["std"]; AMP = True
S = 256
mean = torch.tensor(cfgm, device=dev).view(1, 3, 1, 1); std = torch.tensor(cfgs, device=dev).view(1, 3, 1, 1)
plain = lambda im: T.functional.to_tensor(im.convert("RGB").resize((S, S), Image.BICUBIC))
aug = T.Compose([T.RandomResizedCrop(S, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(),
                 T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.6), T.ToTensor()])
@torch.no_grad()
def emb(paths, tf=plain, tta=True):
    fs = []
    for i in range(0, len(paths), 16):
        x = torch.stack([tf(Image.open(p)) for p in paths[i:i + 16]]).to(dev)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            f = m(x) + (m(torch.flip(x, [3])) if tta else 0)
        fs.append(f.float().cpu())
    return torch.cat(fs).numpy()
rows = list(csv.DictReader(open("data/train-output.csv")))
rp = ["data/train-input/" + r["filename"] for r in rows]
zi = list(csv.DictReader(open("ext/zappos_index.csv")))
sel = lambda role, split=None: [r for r in zi if r["role"] == role and (split is None or r["split"] == split)]
oe_tr = sel("oe", "train"); random.Random(0).shuffle(oe_tr); oe_tr = oe_tr[:3000]
sets = {"zk_tr": sel("val_known", "train"), "zk_va": sel("val_known", "val"), "zu_va": sel("val_unk"),
        "oe_tr": oe_tr, "oe_va": sel("oe", "val")}
res = {"real": emb(rp), "real_lab": np.array([r["label"] for r in rows]),
       "real_aug": np.stack([emb(rp, aug, tta=False) for _ in range(n_aug)])}
for k, v in sets.items():
    res[k] = emb([r["path"] for r in v]); res[k + "_lab"] = np.array([r["label"] if r["role"] == "val_known" else r["brand"] for r in v])
np.savez_compressed(out, **res); print("saved", out, {k: v.shape for k, v in res.items()})
