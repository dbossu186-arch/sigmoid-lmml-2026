"""PRNU presence maps (forgery-localisation style): for a photo group, sum over photos of the local
correlation between residual W_i and I_i*K (K = LOO fingerprint of a camera). Regions where a camera's
PRNU is present light up; a region carrying a different camera's PRNU shows up as a hole / an island.
usage: localize.py <cache> <split_tx.npz> <out_prefix> [sigma]"""
import sys, os, glob
import numpy as np, torch
args = sys.argv[1:]
sys.argv = [sys.argv[0], args[0], "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sp = np.load(args[1]); outp = args[2]; sig = float(args[3]) if len(args) > 3 else 12.0
idx = {(O.who[i], str(O.names[i])): i for i in range(O.N)}
Xi = [idx[("anonymous", n)] for n in sp["X"]] + [idx[("toorisugari", n)] for n in ("w01_012", "w05_006", "w05_013")]
Ti = [idx[("anonymous", n)] for n in sp["T"]] + [idx[("toorisugari", n)] for n in O.names[O.who == "toorisugari"]
                                                   if str(n) not in ("w01_012", "w05_006", "w05_013", "w06_001")]
others = {p: list(np.where(O.who == p)[0]) for p in ["amayadori", "jihanki", "rojiura", "shuuden", "yakin"]}


def gk(s):
    r = int(3 * s); x = torch.arange(-r, r + 1, dtype=torch.float32, device=O.dev); k = torch.exp(-x * x / (2 * s * s)); return k / k.sum()


K1 = gk(sig)


def blur(x):
    p = len(K1) // 2
    x = x[None, None]
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), K1.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), K1.view(1, 1, -1, 1))
    return x[0, 0]


def build(ids):
    num = torch.zeros(O.H, O.W, device=O.dev); den = torch.zeros_like(num); parts = {}
    for i in ids:
        a, b = O.contrib(i); num += a; den += b; parts[i] = (a, b)
    return num, den, parts


def K_for(i, nd):
    num, den, parts = nd
    if i in parts:
        a, b = parts[i]; return O.zm((num - a) / (den - b + 1.0))
    return O.zm(num / (den + 1.0))


fps = {"T": build(Ti), "X": build(Xi)}
groups = {"Tphotos": Ti, "Xphotos": Xi}
res = {}
for gname, ids in groups.items():
    for kname, nd in fps.items():
        SN = torch.zeros(O.H, O.W, device=O.dev); SA = torch.zeros_like(SN); SB = torch.zeros_like(SN)
        for i in ids:
            r = O.R[i].to(O.dev).float(); I = O.I[i].to(O.dev).float(); w = O.Wt[i].to(O.dev).float().sqrt()
            a = r * w; b = I * K_for(i, nd) * w
            SN += blur(a * b); SA += blur(a * a); SB += blur(b * b)
        C = (SN / torch.sqrt(SA * SB + 1e-12)).cpu().numpy()
        res[(gname, kname)] = C
        np.save(f"{outp}_{gname}_K{kname}.npy", C.astype(np.float32))
        print(gname, "vs K", kname, "mean %.4f  p1 %.4f  p99 %.4f" % (C.mean(), np.percentile(C, 1), np.percentile(C, 99)), flush=True)
fig, ax = plt.subplots(2, 2, figsize=(26, 19))
for a, ((g, k), C) in zip(ax.flat, res.items()):
    a.imshow(C, cmap="magma", vmin=np.percentile(C, 1), vmax=np.percentile(C, 99.5)); a.set_title(f"{g} correlated with K_{k}  (sigma {sig})"); a.axis("off")
plt.tight_layout(); plt.savefig(outp + ".png", dpi=40)
