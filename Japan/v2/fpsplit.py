"""Fingerprints of chosen subsets (orientation-aligned), their correlation, and renders.

    python v2/fpsplit.py work work/gattrib_cc.json toorisugari work/fp
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

work, attrib, cam, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
names = json.load(open(work / "meta.json"))["names"]
W = np.load(work / "W.npy", mmap_mode="r"); I = np.load(work / "I.npy", mmap_mode="r")
res = json.load(open(attrib)); mem = res["cams"][cam]
T = lambda a, o: a if o == 0 else a[::-1, ::-1]

def fp(keys):
    num = 0; den = 0
    for k in keys:
        i = names.index(k); o = mem[k]
        w = W[i].astype(np.float32); x = I[i].astype(np.float32)
        num = num + T(w * x, o); den = den + T(x * x, o)
    K = num / (den + 1.0)
    return K - K.mean(), den

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

def render(a, path, sigma=0):
    if sigma: a = ndimage.gaussian_filter(a, sigma)
    lo, hi = np.percentile(a, [1, 99])
    Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)).save(path)

cred = [k for k in mem if not k.startswith("anonymous")]
anon = [k for k in mem if k.startswith("anonymous")]
Kc, dc = fp(cred); Ka, da = fp(anon)
print(cam, "credited", len(cred), "anon", len(anon), "ncc(K_cred, K_anon) = %.4f" % ncc(Kc, Ka))
np.save(out / f"{cam}_Kcred.npy", Kc); np.save(out / f"{cam}_Kanon.npy", Ka)
for tag, K in (("Kcred", Kc), ("Kanon", Ka)):
    for s in (0, 2, 6, 16):
        render(K, out / f"{cam}_{tag}_s{s}.png", s)
    render(K * K, out / f"{cam}_{tag}_energy_s8.png", 8)
D = Ka - Kc * (Ka * Kc).sum() / (Kc * Kc).sum()
for s in (2, 6, 16):
    render(D, out / f"{cam}_D_s{s}.png", s)
render(D * D, out / f"{cam}_D_energy_s8.png", 8)
