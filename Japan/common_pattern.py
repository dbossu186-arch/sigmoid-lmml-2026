"""Whitened common-pattern detector for a group of photos.
   ratio(x) = blur( (sum_i a_i)^2 ) / blur( sum_i a_i^2 ),  a_i = residual_i * I_i * mask_i
   Independent per-photo content (grain, scene leakage) -> ratio ~ 1 everywhere;
   a pattern shared by the group (PRNU, or anything embedded in it) -> ratio > 1 there.
usage: common_pattern.py <cache> <attrib.json> <out_dir> [min_best] [blur]"""
import sys, os, json, glob
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cache, attj, out = sys.argv[1:4]
minb = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
bl = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
os.makedirs(out, exist_ok=True)
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
att = json.load(open(attj))["anon"]
groups = {}
for p in people:
    groups[f"cred_{p}"] = sorted(glob.glob(f"{cache}/{p}__*.npz"))
    groups[f"anon_{p}"] = [f"{cache}/anonymous__{r['name']}.npz" for r in att if r["pred"] == p and r["best"] >= minb]
groups["anon_low"] = [f"{cache}/anonymous__{r['name']}.npz" for r in att if r["best"] < minb]
groups["anon_all"] = [f"{cache}/anonymous__{r['name']}.npz" for r in att]
for wk in range(8, 13):
    groups[f"week{wk:02d}"] = sorted(glob.glob(f"{cache}/anonymous__w{wk:02d}_*.npz"))
only = os.environ.get("GROUPS")
if only:
    groups = {k: v for k, v in groups.items() if k in only.split(",")}

# global mean pattern over ALL photos (pipeline artifacts: residual/intensity bias, JPEG grid)
Mall = None; Ncount = 0
for f in sorted(glob.glob(f"{cache}/*.npz")):
    d = np.load(f); a = d["r"].astype(np.float32) * d["I"].astype(np.float32) / 128.0
    Mall = a if Mall is None else Mall + a; Ncount += 1
Mall /= Ncount
SUB = os.environ.get("SUBK")  # optional: subtract I*K of a credited person (e.g. toorisugari) -> look for a SECOND pattern
Ksub = np.load(SUB) if SUB else None

for g, fs in groups.items():
    S = None; E = None
    for f in fs:
        d = np.load(f)
        r = d["r"].astype(np.float32); I = d["I"].astype(np.float32)
        a = r * I / 128.0 - Mall
        if Ksub is not None:
            a = a - (I * I / 128.0) * Ksub
        S = a if S is None else S + a
        E = a * a if E is None else E + a * a
    ratio = ndimage.gaussian_filter(S * S, bl) / (ndimage.gaussian_filter(E, bl) + 1e-6)
    np.save(f"{out}/ratio_{g}.npy", ratio.astype(np.float32)); np.save(f"{out}/S_{g}.npy", S.astype(np.float32))
    v = ratio[E > np.percentile(E, 5)]
    print(f"{g:18s} n={len(fs):3d} ratio mean {v.mean():.3f} p99 {np.percentile(v, 99):.3f} max {v.max():.2f}", flush=True)
    plt.figure(figsize=(16, 12))
    plt.imshow(ratio, cmap="gray", vmin=np.percentile(ratio, 1), vmax=np.percentile(ratio, 99.5))
    plt.title(f"{g} n={len(fs)} common-pattern ratio (blur {bl})"); plt.axis("off"); plt.tight_layout()
    plt.savefig(f"{out}/ratio_{g}.png", dpi=50); plt.close()
