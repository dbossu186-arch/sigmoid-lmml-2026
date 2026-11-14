"""Per-week local agreement maps for one camera's anonymous photos.

    python v2/weekmaps.py work work/gattrib_cc.json toorisugari work/weekmaps
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

work, cam, out = Path(sys.argv[1]), sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
names = json.load(open(work / "meta.json"))["names"]
W = np.load(work / "W.npy", mmap_mode="r"); I = np.load(work / "I.npy", mmap_mode="r")
res = json.load(open(sys.argv[2]))
T = lambda a, o: a if o == 0 else a[::-1, ::-1]
rows = sorted([r for r in res["anonymous"] if r["pred"] == cam], key=lambda r: r["name"])

def xs(r, frame):
    i = names.index(r["name"])
    x = W[i].astype(np.float32) * I[i].astype(np.float32)
    x /= np.sqrt((x * x).mean()) + 1e-9
    return T(x, r["ori"]) if frame == "sensor" else x

def norm8(a):
    lo, hi = np.percentile(a, [1, 99.5])
    return (np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)

for frame in ("sensor", "image"):
    tiles = []
    for wk in range(8, 13):
        sel = [r for r in rows if int(r["name"].split("__w")[1][:2]) == wk]
        F = 0; E = 0
        for r in sel:
            x = xs(r, frame); F = F + x; E = E + x * x
        for s in (6, 12):
            a = ndimage.gaussian_filter(F * F - E, s) / (ndimage.gaussian_filter(E, s) + 1e-6)
            np.save(out / f"{cam}_w{wk}_{frame}_s{s}.npy", a.astype(np.float32))
        tiles.append(Image.fromarray(norm8(np.load(out / f"{cam}_w{wk}_{frame}_s12.npy"))).resize((640, 480)))
        print(frame, wk, len(sel), flush=True)
    sheet = Image.new("L", (640 * 3, 480 * 2))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 3) * 640, (k // 3) * 480))
    sheet.save(out / f"{cam}_weeks_{frame}.png")
