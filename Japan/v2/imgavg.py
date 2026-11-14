"""Image-frame z-maps (plain averages, no orientation alignment) of photo groups, minus the
all-photo common pattern estimated from the OTHER groups.

    python v2/imgavg.py work work/gattrib_white.json work/imgavg
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

work, out = Path(sys.argv[1]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
names = json.load(open(work / "meta.json"))["names"]
W = np.load(work / "W_white.npy", mmap_mode="r"); I = np.load(work / "I.npy", mmap_mode="r")
res = json.load(open(sys.argv[2]))
pred = {r["name"]: r["pred"] for r in res["anonymous"]}
grp = {}
for k in names:
    who = k.split("__")[0]
    g = f"anon_{pred[k]}" if who == "anonymous" else f"cred_{who}"
    grp.setdefault(g, []).append(names.index(k))
S = {}; V = {}
for g, idx in grp.items():
    s = 0; v = 0
    for i in idx:
        a = W[i].astype(np.float32) * I[i].astype(np.float32); s = s + a; v = v + a * a
    S[g] = s; V[g] = v
St = sum(S.values()); Vt = sum(V.values())
def norm8(a):
    l, h = np.percentile(a, [1, 99]); return (np.clip((a - l) / (h - l + 1e-12), 0, 1) * 255).astype(np.uint8)
tiles = []
for g in sorted(grp):
    n = len(grp[g]); N = sum(len(v) for v in grp.values())
    common = (St - S[g]) / (N - n)  # mean of the other photos
    d = S[g] - n * common
    z = d / np.sqrt(V[g] + n * n * (Vt - V[g]) / (N - n) ** 2 + 1e-6)
    np.save(out / f"z_{g}.npy", z.astype(np.float32))
    for s in (1.5, 4):
        Image.fromarray(norm8(ndimage.gaussian_filter(z, s))).save(out / f"z_{g}_s{s}.png")
    im = Image.fromarray(norm8(ndimage.gaussian_filter(z, 4))).resize((480, 360)); ImageDraw.Draw(im).text((4, 4), g, fill=255)
    tiles.append(im)
    print(g, n, "z std %.3f" % z.std(), flush=True)
sheet = Image.new("L", (4 * 480, 3 * 360))
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % 4) * 480, (k // 4) * 360))
sheet.save(out / "all_s4.png")
