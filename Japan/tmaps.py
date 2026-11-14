"""Per-pixel Welch t-maps between photo groups at several spatial bands (relative to local brightness).
A pattern shared by group A but not B (at any scale, any sign) shows up as |t| >> 1; scene content
is independent between photos and averages out.
usage: tmaps.py <dataset> <attrib.json> <out_dir> <minbest>"""
import sys, os, json, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P
from viz_fp import gblur, dev
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

data, attj, out = sys.argv[1:4]; minb = float(sys.argv[4])
os.makedirs(out, exist_ok=True)
att = json.load(open(attj))["anon"]
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
BANDS = [(0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 24.0), (24.0, 64.0)]

paths, tags = [], []
for p in people:
    for f in sorted(glob.glob(f"{data}/credited/{p}/*.jpg")):
        paths.append(f); tags.append(("cred", p))
for r in att:
    paths.append(f"{data}/anonymous/{r['name']}.jpg")
    tags.append(("anon", r["pred"] if r["best"] >= minb else "low"))

acc = {}  # group -> [sum, sumsq, count] per band


def add(key, vals, m):
    if key not in acc:
        acc[key] = [[torch.zeros_like(v) for v in vals], [torch.zeros_like(v) for v in vals], torch.zeros_like(m)]
    s, q, c = acc[key]
    for b, v in enumerate(vals):
        s[b] += v * m; q[b] += v * v * m
    c += m


for i, (p, (kind, who)) in enumerate(zip(paths, tags)):
    rgb = P.load_rgb(p); m = torch.from_numpy(P.caption_mask(rgb)).to(dev)
    g = torch.from_numpy(rgb.mean(2)).to(dev)[None, None]
    base = gblur(g, 64)[0, 0] + 5.0
    vals = []
    for a, b in BANDS:
        lo = g if a == 0 else gblur(g, a)
        v = (lo - gblur(g, b))[0, 0] / base          # relative (multiplicative) band component
        vals.append(v)
    add(f"{kind}_{who}", vals, m)
    add(f"{kind}_all", vals, m)

stats = {}
def tmap(A, B):
    sA, qA, cA = acc[A]; sB, qB, cB = acc[B]
    res = []
    for b in range(len(BANDS)):
        mA = sA[b] / cA.clamp(min=1); vA = qA[b] / cA.clamp(min=1) - mA ** 2
        mB = sB[b] / cB.clamp(min=1); vB = qB[b] / cB.clamp(min=1) - mB ** 2
        t = (mA - mB) / torch.sqrt(vA / cA.clamp(min=1) + vB / cB.clamp(min=1) + 1e-12)
        t[(cA < 3) | (cB < 3)] = 0
        res.append(t.cpu().numpy())
    return res

pairs = [("anon_toorisugari", "anon_all"), ("cred_toorisugari", "cred_all"), ("anon_toorisugari", "cred_toorisugari"),
         ("anon_yakin", "anon_all"), ("anon_jihanki", "anon_all"), ("anon_low", "anon_all"), ("anon_all", "cred_all")]
for A, B in pairs:
    if A not in acc or B not in acc:
        continue
    ts = tmap(A, B)
    fig, ax = plt.subplots(2, len(BANDS), figsize=(8 * len(BANDS), 12))
    for k, t in enumerate(ts):
        tb = t
        ax[0, k].imshow(tb, cmap="gray", vmin=-4, vmax=4); ax[0, k].set_title(f"{A} vs {B} band {BANDS[k]} t"); ax[0, k].axis("off")
        sm = __import__("scipy.ndimage", fromlist=["x"]).gaussian_filter(t, 6) * 6
        ax[1, k].imshow(sm, cmap="gray", vmin=-4, vmax=4); ax[1, k].set_title("smoothed"); ax[1, k].axis("off")
        np.save(f"{out}/t_{A}_vs_{B}_b{k}.npy", t.astype(np.float32))
        print(f"{A:18s} vs {B:16s} band {BANDS[k]} std(t)={t[t != 0].std():.3f} max|t|={np.abs(t).max():.1f}", flush=True)
    plt.tight_layout(); plt.savefig(f"{out}/t_{A}_vs_{B}.png", dpi=40); plt.close()
print({k: int(v[2].max().item()) for k, v in acc.items()})
