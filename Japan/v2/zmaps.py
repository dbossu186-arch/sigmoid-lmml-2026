"""Per-camera fingerprint z-maps from whitened residuals (orientation-aligned, all members).

z_c = sum_i T(W_i I_i) / sqrt(sum_i T(W_i^2 I_i^2)) : unit-variance under 'no fingerprint'.
Saves z maps and local-energy renders (smooth(z^2) - 1).

    python v2/zmaps.py work work/gattrib_white.json work/zmaps
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

work, out = Path(sys.argv[1]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
names = json.load(open(work / "meta.json"))["names"]
W = np.load(work / "W_white.npy", mmap_mode="r"); I = np.load(work / "I.npy", mmap_mode="r")
res = json.load(open(sys.argv[2]))
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
T = lambda a, o: a if o == 0 else a[::-1, ::-1]
mem = {p: dict(res["cams"][p]) for p in people}
for r in res["anonymous"]:
    mem[r["pred"]].setdefault(r["name"], r["ori"])

def norm8(a, lo=1, hi=99):
    l, h = np.percentile(a, [lo, hi]); return (np.clip((a - l) / (h - l + 1e-12), 0, 1) * 255).astype(np.uint8)

tiles = {}
for p in people:
    S = 0; V = 0
    for k, o in mem[p].items():
        i = names.index(k); a = W[i].astype(np.float32) * I[i].astype(np.float32)
        S = S + T(a, o); V = V + T(a * a, o)
    z = S / np.sqrt(V + 1e-6)
    z[V < np.percentile(V, 2)] = 0
    np.save(out / f"z_{p}.npy", z.astype(np.float32))
    for s in (4, 10):
        e = ndimage.gaussian_filter(z * z, s) - 1
        Image.fromarray(norm8(e, 1, 99.5)).save(out / f"energy_{p}_s{s}.png")
        tiles[(p, s)] = e
    Image.fromarray(norm8(ndimage.gaussian_filter(z, 2))).save(out / f"z_{p}_s2.png")
    print(p, len(mem[p]), "z std %.3f  energy mean %.3f" % (z.std(), (z * z).mean() - 1), flush=True)
for s in (4, 10):
    sheet = Image.new("L", (3 * 640, 2 * 480))
    for n, p in enumerate(people):
        im = Image.fromarray(norm8(tiles[(p, s)], 1, 99.5)).resize((640, 480))
        ImageDraw.Draw(im).text((4, 4), p, fill=255)
        sheet.paste(im, ((n % 3) * 640, (n // 3) * 480))
    sheet.save(out / f"energy_all_s{s}.png")
