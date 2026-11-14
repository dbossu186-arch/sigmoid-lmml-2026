"""Pairwise residual correlation over all photos (credited + anonymous). Saves G matrix + names."""
import sys, glob, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P
from viz_fp import gblur, zm, dev

data, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
variant = sys.argv[3] if len(sys.argv) > 3 else "hp0.5_tex"
paths = sorted(glob.glob(f"{data}/credited/*/*.jpg")) + sorted(glob.glob(f"{data}/anonymous/*.jpg"))
labels = [p.split("/")[-2] for p in paths]
names = [os.path.basename(p)[:-4] for p in paths]

feats = []
for p in paths:
    rgb = P.load_rgb(p); m = torch.from_numpy(P.caption_mask(rgb)).to(dev)[None, None]
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    s = float(variant.split("_")[0][2:])
    r = x - gblur(x, s)
    r = zm(r * m) * m
    if "tex" in variant:
        w = 1.0 / torch.sqrt(gblur(r * r, 4) + 1.0)
    else:
        w = torch.ones_like(r)
    if "noI" in variant:
        f = (r * w)
    else:
        f = r * w  # query side residual ; PRNU appears as I*K -> correlate r with r' weighted by I*I'
        f = f * (x * m) / 128.0
    f = f.mean(1)  # combine channels
    f = f - f.mean()
    f = f / f.norm()
    feats.append(f.flatten().half().cpu())
X = torch.stack(feats).float()
G = (X @ X.T).numpy()
np.save(f"{out}/G_{variant}.npy", G)
json.dump(dict(names=names, labels=labels), open(f"{out}/G_names.json", "w"))
lab = np.array(labels); cred = lab != "anonymous"
off = ~np.eye(len(lab), dtype=bool)
same = (lab[:, None] == lab[None]) & cred[:, None] & cred[None] & off
diff = (lab[:, None] != lab[None]) & cred[:, None] & cred[None]
mu, sd = G[diff].mean(), G[diff].std()
print(variant, "credited same z=%.2f (mean %.5f) diff mean %.5f sd %.5f" % ((G[same].mean() - mu) / sd, G[same].mean(), mu, sd))
