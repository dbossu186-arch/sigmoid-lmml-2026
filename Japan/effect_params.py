"""Per-photo Subtitle-Effect parameters: grain std (flat regions), colour balance, caption geometry, black level.
usage: effect_params.py <dataset> <split_tx.npz> <attrib.json> <out.csv>"""
import sys, os, glob, json, csv
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P

data, spf, attj, outc = sys.argv[1:5]
sp = np.load(spf); X = set(sp["X"]); T = set(sp["T"])
att = {r["name"]: r for r in json.load(open(attj))["anon"]}
Xc = {"w01_012", "w05_006", "w05_013"}
rows = []
for p in sorted(glob.glob(f"{data}/credited/*/*.jpg")) + sorted(glob.glob(f"{data}/anonymous/*.jpg")):
    who = p.split("/")[-2]; n = os.path.basename(p)[:-4]
    if who == "anonymous":
        cam = "X" if n in X else ("T" if n in T else (att[n]["pred"] if att[n]["best"] >= 10 else "low"))
    else:
        cam = "X" if (who == "toorisugari" and n in Xc) else ("T" if who == "toorisugari" else who)
    rgb = P.load_rgb(p); m = P.caption_mask(rgb) > 0; g = rgb.mean(2)
    r = g - ndimage.gaussian_filter(g, 1.0)
    tex = ndimage.uniform_filter(np.abs(g - ndimage.gaussian_filter(g, 3)), 15)
    flat = m & (tex < np.percentile(tex[m], 15)) & (g > 30) & (g < 220)
    grain = r[flat].std()
    R, G, B = (rgb[..., c][m].mean() for c in range(3))
    capmask = P.caption_mask(rgb) == 0
    rowsdark = (g < 10).mean(1) > 0.6
    top = int(rowsdark[:480].sum()); bot = int(rowsdark[480:].sum())
    black = np.percentile(g[m], 0.5); white = np.percentile(g[m], 99.5)
    rows.append(dict(cam=cam, who=who, name=n, grain=round(float(grain), 3), RG=round(R / G, 4), BG=round(B / G, 4),
                     mean=round(float(g[m].mean()), 1), bar_top=top, bar_bot=bot, black=round(float(black), 1), white=round(float(white), 1),
                     masked=round(float(capmask.mean()), 3)))
with open(outc, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import collections
by = collections.defaultdict(list)
for r_ in rows:
    by[r_["cam"]].append(r_)
for cam, L in sorted(by.items()):
    print(f"{cam:12s} n={len(L):3d} " + " ".join(f"{k}={np.mean([x[k] for x in L]):.3f}±{np.std([x[k] for x in L]):.3f}" for k in ["grain", "RG", "BG", "mean", "black", "bar_top", "bar_bot"]))
