"""Render group fingerprints (wav residual, global-mean removed) at several scales.
usage: show_K.py <cache> <attrib.json> <minbest> <out_png> group[,group...]
group syntax: anon:<person> | cred:<person> | both:<person>"""
import sys, glob, json
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cache, attj, minb, outp, groups = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5].split(",")
att = json.load(open(attj))["anon"]
allf = sorted(glob.glob(f"{cache}/*.npz"))
M = 0
for f in allf:
    d = np.load(f); M = M + d["r"].astype(np.float32) * d["I"].astype(np.float32)
M = M / len(allf)


def files(g):
    kind, p = g.split(":")
    an = [f"{cache}/anonymous__{r['name']}.npz" for r in att if r["pred"] == p and r["best"] >= minb]
    cr = sorted(glob.glob(f"{cache}/{p}__*.npz"))
    return {"anon": an, "cred": cr, "both": an + cr}[kind]


def K(fs):
    num = 0; den = 0
    for f in fs:
        d = np.load(f); r = d["r"].astype(np.float32); I = d["I"].astype(np.float32); w = d["w"].astype(np.float32)
        num = num + (r * I - M) * w; den = den + I * I * w
    k = num / (den + 50.0)
    k -= k.mean(0, keepdims=True); k -= k.mean(1, keepdims=True)
    return k


scales = [0, 2, 5, 10]
fig, ax = plt.subplots(len(groups), len(scales) + 1, figsize=(7 * (len(scales) + 1), 5.4 * len(groups)))
ax = np.atleast_2d(ax)
for gi, g in enumerate(groups):
    fs = files(g); k = K(fs)
    np.save(outp.replace(".png", f"_{g.replace(':', '_')}.npy"), k.astype(np.float32))
    for si, s in enumerate(scales):
        v = ndimage.gaussian_filter(k, s) if s else k
        ax[gi, si].imshow(v, cmap="gray", vmin=np.percentile(v, 1), vmax=np.percentile(v, 99)); ax[gi, si].set_title(f"{g} n={len(fs)} blur {s}")
        ax[gi, si].axis("off")
    e = ndimage.gaussian_filter(k * k, 6)
    ax[gi, -1].imshow(e, cmap="gray", vmin=np.percentile(e, 1), vmax=np.percentile(e, 99)); ax[gi, -1].set_title("local energy"); ax[gi, -1].axis("off")
plt.tight_layout(); plt.savefig(outp, dpi=40)
