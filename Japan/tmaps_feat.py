"""Group t-maps for many per-pixel features (chroma, per-channel, band-passed), to hunt a hidden pattern
in a specific channel. usage: tmaps_feat.py <dataset> <attrib.json> <out_dir> <minbest> [groupA,groupB ...]"""
import sys, os, json, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P
from viz_fp import gblur, dev
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

data, attj, out = sys.argv[1:4]; minb = float(sys.argv[4])
os.makedirs(out, exist_ok=True)
att = json.load(open(attj))["anon"]
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
paths, tags = [], []
for p in people:
    for f in sorted(glob.glob(f"{data}/credited/{p}/*.jpg")):
        paths.append(f); tags.append(("cred", p))
for r in att:
    paths.append(f"{data}/anonymous/{r['name']}.jpg"); tags.append(("anon", r["pred"] if r["best"] >= minb else "low"))


def feats(rgb):
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    R, G, B = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
    f = {}
    for n, c in (("Cb", Cb), ("Cr", Cr)):
        f[n + "_hp2"] = (c - gblur(c, 2.0))
        f[n + "_bp"] = (gblur(c, 2.0) - gblur(c, 12.0))
        f[n + "_raw"] = c - gblur(c, 40.0)
    for n, c in (("R", R), ("G", G), ("B", B)):
        f[n + "_hp1"] = (c - gblur(c, 1.0)) / (gblur(Y, 30) + 5)
    f["RmB_hp1"] = f["R_hp1"] - f["B_hp1"]
    return {k: v[0, 0] for k, v in f.items()}


acc = {}
for p, (kind, who) in zip(paths, tags):
    rgb = P.load_rgb(p); m = torch.from_numpy(P.caption_mask(rgb)).to(dev)
    fs = feats(rgb)
    for key in (f"{kind}_{who}", f"{kind}_all"):
        if key not in acc:
            acc[key] = ({k: torch.zeros_like(v) for k, v in fs.items()}, {k: torch.zeros_like(v) for k, v in fs.items()}, torch.zeros_like(m))
        s, q, c = acc[key]
        for k, v in fs.items():
            s[k] += v * m; q[k] += v * v * m
        c += m

pairs = [a.split(",") for a in sys.argv[5:]] or [["anon_toorisugari", "anon_all"], ["cred_toorisugari", "cred_all"], ["anon_all", "cred_all"]]
for A, B in pairs:
    sA, qA, cA = acc[A]; sB, qB, cB = acc[B]
    keys = list(sA)
    fig, ax = plt.subplots(3, (len(keys) + 2) // 3, figsize=(7 * ((len(keys) + 2) // 3), 16))
    for a_, k in zip(ax.flat, keys):
        mA = sA[k] / cA.clamp(min=1); vA = qA[k] / cA.clamp(min=1) - mA ** 2
        mB = sB[k] / cB.clamp(min=1); vB = qB[k] / cB.clamp(min=1) - mB ** 2
        t = ((mA - mB) / torch.sqrt(vA / cA.clamp(min=1) + vB / cB.clamp(min=1) + 1e-12)).cpu().numpy()
        t[((cA < 3) | (cB < 3)).cpu().numpy()] = 0
        ts = ndimage.gaussian_filter(t, 3) * 3
        a_.imshow(ts, cmap="gray", vmin=-4, vmax=4); a_.set_title(f"{k} std={t[t != 0].std():.2f} max={np.abs(t).max():.1f}"); a_.axis("off")
        print(f"{A} vs {B} {k:8s} std(t)={t[t != 0].std():.3f} smoothed std={ts.std():.3f} max|t|={np.abs(t).max():.1f}", flush=True)
    plt.suptitle(f"{A} vs {B}"); plt.tight_layout(); plt.savefig(f"{out}/tf_{A}_vs_{B}.png", dpi=40); plt.close()
