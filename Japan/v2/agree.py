"""Local agreement maps: where do the photos of a set share residual content?

A(x,y) = smooth(F^2 - sum x_i^2) / smooth(sum x_i^2),  F = sum_i T_i(x_i),  x_i = W_i * I_i
(sum over pairs of local products, normalized by local energy).

    python v2/agree.py work work/gattrib_cc.json work/agree
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

work, out = Path(sys.argv[1]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
names = json.load(open(work / "meta.json"))["names"]
W = np.load(work / "W.npy", mmap_mode="r"); I = np.load(work / "I.npy", mmap_mode="r")
res = json.load(open(sys.argv[2]))
T = lambda a, o: a if o == 0 else a[::-1, ::-1]

def amap(keys, frame, sigma=8):
    F = 0; E = 0
    for k, o in keys:
        i = names.index(k)
        x = W[i].astype(np.float32) * I[i].astype(np.float32)
        x /= np.sqrt((x * x).mean()) + 1e-9
        x = T(x, o) if frame == "sensor" else x
        F = F + x; E = E + x * x
    return ndimage.gaussian_filter(F * F - E, sigma) / (ndimage.gaussian_filter(E, sigma) + 1e-6)

def save(a, path):
    lo, hi = np.percentile(a, [1, 99.5])
    Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)).save(path)

for cam in ("toorisugari", "yakin", "rojiura"):
    mem = res["cams"][cam]
    sets = {"anon": [(k, o) for k, o in mem.items() if k.startswith("anonymous")],
            "cred": [(k, o) for k, o in mem.items() if not k.startswith("anonymous")]}
    for tag, keys in sets.items():
        for frame in ("sensor", "image"):
            a = amap(keys, frame)
            np.save(out / f"{cam}_{tag}_{frame}.npy", a.astype(np.float32))
            save(a, out / f"{cam}_{tag}_{frame}.png")
            print(cam, tag, frame, len(keys), "mean %.3f std %.3f max %.3f" % (a.mean(), a.std(), a.max()), flush=True)
