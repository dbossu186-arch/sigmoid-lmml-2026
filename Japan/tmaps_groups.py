"""Welch t-maps between an explicit photo group and the rest, for several bands (relative to brightness).
usage: tmaps_groups.py <dataset> <out_png> <groupA: who/name,...> [groupB or 'rest']"""
import sys, os, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P
from viz_fp import gblur, dev
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

data, outp = sys.argv[1], sys.argv[2]
A = set(sys.argv[3].split(","))
allp = sorted(glob.glob(f"{data}/credited/*/*.jpg")) + sorted(glob.glob(f"{data}/anonymous/*.jpg"))
key = lambda p: p.split("/")[-2] + "/" + os.path.basename(p)[:-4]
B = set(sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] != "rest" else {key(p) for p in allp} - A
BANDS = [(0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 24.0), (0, 24.0)]
acc = {"A": None, "B": None}
for p in allp:
    k = key(p)
    g = "A" if k in A else ("B" if k in B else None)
    if g is None:
        continue
    rgb = P.load_rgb(p); m = torch.from_numpy(P.caption_mask(rgb)).to(dev)
    x = torch.from_numpy(rgb.mean(2)).to(dev)[None, None]
    vals = [((x if a == 0 else gblur(x, a)) - gblur(x, b))[0, 0] for a, b in BANDS]
    if acc[g] is None:
        acc[g] = [[torch.zeros_like(v) for v in vals], [torch.zeros_like(v) for v in vals], torch.zeros_like(m)]
    s, q, c = acc[g]
    for i, v in enumerate(vals):
        s[i] += v * m; q[i] += v * v * m
    c += m
sA, qA, cA = acc["A"]; sB, qB, cB = acc["B"]
fig, ax = plt.subplots(2, len(BANDS), figsize=(8 * len(BANDS), 12))
for i in range(len(BANDS)):
    mA = sA[i] / cA.clamp(min=1); vA = qA[i] / cA.clamp(min=1) - mA ** 2
    mB = sB[i] / cB.clamp(min=1); vB = qB[i] / cB.clamp(min=1) - mB ** 2
    t = ((mA - mB) / torch.sqrt(vA / cA.clamp(min=1) + vB / cB.clamp(min=1) + 1e-12)).cpu().numpy()
    t[((cA < 3) | (cB < 3)).cpu().numpy()] = 0
    ax[0, i].imshow(t, cmap="gray", vmin=-4, vmax=4); ax[0, i].set_title(f"t band {BANDS[i]} std {t[t!=0].std():.2f} max {np.abs(t).max():.1f}"); ax[0, i].axis("off")
    mm = mA.cpu().numpy(); ax[1, i].imshow(mm, cmap="gray", vmin=np.percentile(mm, 1), vmax=np.percentile(mm, 99)); ax[1, i].set_title("group A mean"); ax[1, i].axis("off")
    np.save(outp.replace(".png", f"_t{i}.npy"), t.astype(np.float32))
    print(f"band {BANDS[i]} std(t)={t[t!=0].std():.3f} max|t|={np.abs(t).max():.1f} n_A={int(cA.max())} n_B={int(cB.max())}", flush=True)
plt.tight_layout(); plt.savefig(outp, dpi=40)
