"""Build a fingerprint from an explicit photo list and render several views.
usage: render_K.py <cache> <out_prefix> <who__name,...>   (who = anonymous|toorisugari|...)"""
import sys, glob
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cache, outp, lst = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
allf = sorted(glob.glob(f"{cache}/*.npz"))
M = 0
for f in allf:
    d = np.load(f); M = M + d["r"].astype(np.float32) * d["I"].astype(np.float32)
M = M / len(allf)
num = 0; den = 0
for n in lst:
    d = np.load(f"{cache}/{n}.npz"); r = d["r"].astype(np.float32); I = d["I"].astype(np.float32); w = d["w"].astype(np.float32)
    num = num + (r * I - M) * w; den = den + I * I * w
k = num / (den + 50.0)
k -= k.mean(0, keepdims=True); k -= k.mean(1, keepdims=True)
np.save(outp + "_K.npy", k.astype(np.float32))
views = []
for s in (0, 1.5, 4, 10):
    v = ndimage.gaussian_filter(k, s) if s else k
    views.append((f"K blur {s}", v))
e = ndimage.gaussian_filter(k * k, 6); views.append(("local energy s6", e))
z = k / (np.sqrt(ndimage.gaussian_filter(k * k, 25)) + 1e-9)
views.append(("whitened blur 3", ndimage.gaussian_filter(z, 3)))
fig, ax = plt.subplots(2, 3, figsize=(30, 15))
for a, (t, v) in zip(ax.flat, views):
    a.imshow(v, cmap="gray", vmin=np.percentile(v, 1), vmax=np.percentile(v, 99)); a.set_title(f"{t} n={len(lst)}"); a.axis("off")
plt.tight_layout(); plt.savefig(outp + ".png", dpi=40)
print("saved", outp)
