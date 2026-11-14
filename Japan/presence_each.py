"""Per-photo PRNU presence maps (local correlation with LOO fingerprint), rendered as a grid.
usage: presence_each.py <cache> <split_tx.npz> <out_png> <group: X|T|list> <fp: X|T> [sigma]"""
import sys, os
import numpy as np, torch
args = sys.argv[1:]
sys.argv = [sys.argv[0], args[0], "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sp = np.load(args[1]); outp = args[2]; grp = args[3]; fpn = args[4]; sig = float(args[5]) if len(args) > 5 else 10.0
idx = {(O.who[i], str(O.names[i])): i for i in range(O.N)}
Xc = ("w01_012", "w05_006", "w05_013")
Xi = [idx[("anonymous", n)] for n in sp["X"]] + [idx[("toorisugari", n)] for n in Xc]
Ti = [idx[("anonymous", n)] for n in sp["T"]] + [idx[("toorisugari", str(n))] for n in O.names[O.who == "toorisugari"] if str(n) not in Xc + ("w06_001",)]
fp_ids = {"X": Xi, "T": Ti}[fpn]
if grp in ("X", "T"):
    show = {"X": Xi, "T": Ti}[grp]
else:
    show = [idx[tuple(s.split("/"))] for s in grp.split(",")]
num = torch.zeros(O.H, O.W, device=O.dev); den = torch.zeros_like(num); parts = {}
for i in fp_ids:
    a, b = O.contrib(i); num += a; den += b; parts[i] = (a, b)
r_ = int(3 * sig); x = torch.arange(-r_, r_ + 1, dtype=torch.float32, device=O.dev); g = torch.exp(-x * x / (2 * sig * sig)); g /= g.sum()


def blur(z):
    p = len(g) // 2; z = z[None, None]
    z = F.conv2d(F.pad(z, (p, p, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
    z = F.conv2d(F.pad(z, (0, 0, p, p), mode="reflect"), g.view(1, 1, -1, 1)); return z[0, 0]


n = len(show); cols = 5; rows = (n + cols - 1) // cols
fig, ax = plt.subplots(rows, cols, figsize=(cols * 6.4, rows * 4.8))
for k, i in enumerate(show):
    if i in parts:
        a_, b_ = parts[i]; K = O.zm((num - a_) / (den - b_ + 1.0))
    else:
        K = O.zm(num / (den + 1.0))
    r = O.R[i].to(O.dev).float(); I = O.I[i].to(O.dev).float(); w = O.Wt[i].to(O.dev).float().sqrt()
    a = r * w; b = I * K * w
    C = (blur(a * b) / torch.sqrt(blur(a * a) * blur(b * b) + 1e-12)).cpu().numpy()
    pc = O.pce(i, K)
    axx = ax.flat[k]; axx.imshow(C, cmap="magma", vmin=-0.1, vmax=0.25); axx.set_title(f"{O.who[i][:5]}/{O.names[i]} PCE {pc:.0f}"); axx.axis("off")
for k in range(n, rows * cols):
    ax.flat[k].axis("off")
plt.tight_layout(); plt.savefig(outp, dpi=35)
print("saved", outp)
