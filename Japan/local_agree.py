"""Local agreement map between two disjoint half-fingerprints of a group.
C(x) = blur(KA*KB) / sqrt(blur(KA^2) blur(KB^2)); a pattern shared only in some region (e.g. letters) lights up.
usage: local_agree.py <cache> <attrib.json> <minbest> <out_png> <spec> [<spec> ...]
spec: anon:<person> | cred:<person> | both:<person> | animcred:<person> (anon half vs credited half)"""
import sys, glob, json
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cache, attj, minb, outp = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
specs = sys.argv[5:]
att = json.load(open(attj))["anon"]
allf = sorted(glob.glob(f"{cache}/*.npz"))
M = 0
for f in allf:
    d = np.load(f); M = M + d["r"].astype(np.float32) * d["I"].astype(np.float32)
M = M / len(allf)


def K(fs):
    num = 0; den = 0
    for f in fs:
        d = np.load(f); r = d["r"].astype(np.float32); I = d["I"].astype(np.float32); w = d["w"].astype(np.float32)
        num = num + (r * I - M) * w; den = den + I * I * w
    k = num / (den + 50.0)
    k -= k.mean(0, keepdims=True); k -= k.mean(1, keepdims=True)
    k -= ndimage.gaussian_filter(k, 6)  # drop low-frequency leakage/banding
    return k


rng = np.random.default_rng(1)
fig, ax = plt.subplots(len(specs), 2, figsize=(26, 9.6 * len(specs)))
ax = np.atleast_2d(ax)
for si, spec in enumerate(specs):
    kind, p = spec.split(":")
    an = [f"{cache}/anonymous__{r['name']}.npz" for r in att if r["pred"] == p and r["best"] >= minb]
    cr = sorted(glob.glob(f"{cache}/{p}__*.npz"))
    if kind == "animcred":
        A, B = an, cr
    else:
        fs = {"anon": an, "cred": cr, "both": an + cr}[kind]
        perm = rng.permutation(len(fs)); A = [fs[i] for i in perm[::2]]; B = [fs[i] for i in perm[1::2]]
    kA, kB = K(A), K(B)
    for j, s in enumerate((8, 16)):
        C = ndimage.gaussian_filter(kA * kB, s) / np.sqrt(ndimage.gaussian_filter(kA * kA, s) * ndimage.gaussian_filter(kB * kB, s) + 1e-12)
        ax[si, j].imshow(C, cmap="magma", vmin=np.percentile(C, 1), vmax=np.percentile(C, 99.5))
        ax[si, j].set_title(f"{spec} |A|={len(A)} |B|={len(B)} blur {s} mean={C.mean():.3f}"); ax[si, j].axis("off")
        np.save(outp.replace(".png", f"_{kind}_{p}_b{s}.npy"), C.astype(np.float32))
        print(spec, s, "mean %.3f std %.3f p99.5 %.3f" % (C.mean(), C.std(), np.percentile(C, 99.5)), flush=True)
plt.tight_layout(); plt.savefig(outp, dpi=40)
