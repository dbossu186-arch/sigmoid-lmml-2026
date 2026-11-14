"""Per-photo local correlation with a camera fingerprint (tampering-localization style).

    python v2/localcorr.py work work/gattrib_cc.json toorisugari work/localcorr [sigma]
For each member i: rho(x,y) = smooth(a*b)/sqrt(smooth(a^2)*smooth(b^2)) with a = T(W_i), b = T(I_i)*K^{-i}.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

work, cam, out = Path(sys.argv[1]), sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
sigma = float(sys.argv[5]) if len(sys.argv) > 5 else 24
names = json.load(open(work / "meta.json"))["names"]
W = np.load(work / "W.npy", mmap_mode="r"); I = np.load(work / "I.npy", mmap_mode="r")
res = json.load(open(sys.argv[2])); mem = res["cams"][cam]
for r in res["anonymous"]:
    if r["pred"] == cam and r["name"] not in mem: mem[r["name"]] = r["ori"]
T = lambda a, o: a if o == 0 else a[::-1, ::-1]
num = 0; den = 0; parts = {}
for k, o in mem.items():
    i = names.index(k); w = W[i].astype(np.float32); x = I[i].astype(np.float32)
    a = T(w * x, o); b = T(x * x, o); num = num + a; den = den + b
maps = {}
for k, o in sorted(mem.items()):
    i = names.index(k); w = W[i].astype(np.float32); x = I[i].astype(np.float32)
    K = (num - T(w * x, o)) / (den - T(x * x, o) + 1.0); K -= K.mean()
    a = T(w, o); b = T(x, o) * K
    g = lambda z: ndimage.gaussian_filter(z, sigma)
    rho = g(a * b) / np.sqrt(g(a * a) * g(b * b) + 1e-12)
    rho = T(rho, o)  # back to image frame so it lines up with the photo
    maps[k] = rho.astype(np.float32)
np.savez_compressed(out / f"{cam}_s{int(sigma)}.npz", **maps)
keys = sorted(maps); cols = 6; th = (320, 240)
for part in range(0, len(keys), 24):
    sub = keys[part:part + 24]; rows = (len(sub) + cols - 1) // cols
    sheet = Image.new("L", (cols * th[0], rows * th[1]))
    for n, k in enumerate(sub):
        a = maps[k]; lo, hi = np.percentile(a, [1, 99])
        im = Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)).resize(th)
        ImageDraw.Draw(im).text((3, 3), k.split("__")[-1], fill=255)
        sheet.paste(im, ((n % cols) * th[0], (n // cols) * th[1]))
    sheet.save(out / f"{cam}_s{int(sigma)}_{part // 24}.png")
print("maps", len(maps))
