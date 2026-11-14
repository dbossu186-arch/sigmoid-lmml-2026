"""Full cross-correlation of each photo with its camera fingerprint (LOO), all circular shifts.
Reports PCE at zero shift and the strongest off-zero peaks.

    python v2/shiftscan.py work work/gattrib_cc.json work/shiftscan.json
"""
import json, sys
from pathlib import Path
import numpy as np

work = Path(sys.argv[1]); names = json.load(open(work / "meta.json"))["names"]
W = np.load(work / "W.npy", mmap_mode="r"); I = np.load(work / "I.npy", mmap_mode="r")
res = json.load(open(sys.argv[2]))
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
T = lambda a, o: a if o == 0 else a[::-1, ::-1]
mem = {p: dict(res["cams"][p]) for p in people}
for r in res["anonymous"]:
    mem[r["pred"]].setdefault(r["name"], r["ori"])
sums = {}
for p in people:
    num = 0; den = 0
    for k, o in mem[p].items():
        i = names.index(k); w = W[i].astype(np.float32); x = I[i].astype(np.float32)
        num = num + T(w * x, o); den = den + T(x * x, o)
    sums[p] = (num, den)
out = []
for p in people:
    num, den = sums[p]
    for k, o in mem[p].items():
        i = names.index(k); w = W[i].astype(np.float32); x = I[i].astype(np.float32)
        K = (num - T(w * x, o)) / (den - T(x * x, o) + 1.0); K -= K.mean()
        a = T(w, o); b = T(x, o) * K
        a = a - a.mean(); b = b - b.mean()
        xc = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
        H, Wd = xc.shape
        e = xc.copy(); e[:6, :6] = 0; e[-5:, :6] = 0; e[:6, -5:] = 0; e[-5:, -5:] = 0
        energy = (e ** 2).mean()
        pce0 = float(np.sign(xc[0, 0]) * xc[0, 0] ** 2 / energy)
        flat = np.argsort(e.ravel())[::-1][:3]
        peaks = []
        for f in flat:
            y, xx = divmod(int(f), Wd)
            dy = y if y < H // 2 else y - H; dx = xx if xx < Wd // 2 else xx - Wd
            peaks.append([dy, dx, float(e.ravel()[f] ** 2 / energy)])
        out.append({"name": k, "cam": p, "ori": o, "pce0": pce0, "peaks": peaks})
        print(p, k, "pce0 %.1f" % pce0, "top off-zero", [(a_, b_, round(c_, 1)) for a_, b_, c_ in peaks[:2]], flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
