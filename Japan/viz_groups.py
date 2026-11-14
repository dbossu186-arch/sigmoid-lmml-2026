"""Average per-group maps to reveal hidden structure:
   K   : MLE high-pass fingerprint (cache r, I, w)        -> show local energy and blurred K
   FF  : flat-field ratio  gray / blur(gray, 25) - 1      -> low/mid-frequency pattern
usage: viz_groups.py <dataset> <cache> <attrib.json> <out_dir> [min_best]"""
import sys, os, glob, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P
from viz_fp import gblur, zm, dev
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

data, cache, attj, out = sys.argv[1:5]
minb = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
os.makedirs(out, exist_ok=True)
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
att = json.load(open(attj))["anon"]
groups = {}
for p in people:
    groups[f"cred_{p}"] = [("credited", p, os.path.basename(f)[:-4]) for f in sorted(glob.glob(f"{data}/credited/{p}/*.jpg"))]
    groups[f"anon_{p}"] = [("anonymous", p, r["name"]) for r in att if r["pred"] == p and r["best"] >= minb]
groups["anon_low"] = [("anonymous", None, r["name"]) for r in att if r["best"] < minb]
groups["anon_all"] = [("anonymous", None, r["name"]) for r in att]


def load(kind, p, n):
    path = f"{data}/credited/{p}/{n}.jpg" if kind == "credited" else f"{data}/anonymous/{n}.jpg"
    rgb = P.load_rgb(path); m = P.caption_mask(rgb)
    c = np.load(f"{cache}/{'anonymous' if kind == 'anonymous' else p}__{n}.npz")
    return rgb, m, c


res = {}
for g, items in groups.items():
    num = den = ff = ffn = None
    for kind, p, n in items:
        rgb, m, c = load(kind, p, n)
        r = torch.from_numpy(c["r"].astype(np.float32)).to(dev); I = torch.from_numpy(c["I"].astype(np.float32)).to(dev)
        w = torch.from_numpy(c["w"].astype(np.float32)).to(dev)
        a = r * I * w; b = I * I * w
        num = a if num is None else num + a; den = b if den is None else den + b
        gray = torch.from_numpy(rgb.mean(2)).to(dev)[None, None]
        mm = torch.from_numpy(m).to(dev)[None, None]
        bl = gblur(gray * mm, 25) / (gblur(mm, 25) + 1e-3)
        f = ((gray / (bl + 1.0) - 1.0) * mm)[0, 0]
        ff = f if ff is None else ff + f
        ffn = mm[0, 0] if ffn is None else ffn + mm[0, 0]
    K = zm(num / (den + 1.0))
    E = gblur((K * K)[None, None], 5)[0, 0].cpu().numpy()
    Kb = gblur(K[None, None], 2.5)[0, 0].cpu().numpy()
    FF = (ff / (ffn + 1e-3)).cpu().numpy()
    res[g] = (E, Kb, FF, len(items))
    np.save(f"{out}/K_{g}.npy", K.cpu().numpy().astype(np.float32))
    np.save(f"{out}/FF_{g}.npy", FF.astype(np.float32))
    print(g, len(items), flush=True)

for g, (E, Kb, FF, n) in res.items():
    fig, ax = plt.subplots(1, 3, figsize=(27, 7))
    for a_, img, t in zip(ax, (E, Kb, FF), ("energy K", "blur K", "flat-field")):
        a_.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99)); a_.set_title(f"{g} n={n} {t}")
        a_.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/{g}.png", dpi=45); plt.close()
