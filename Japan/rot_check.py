"""Orientation-aware scoring: every photo's residual (id / rot180 / flipLR / flipUD) against every camera
fingerprint built from upright credited photos (Toorisugari credited minus the 3 rotated ones).
usage: rot_check.py <cache> <out.json>"""
import sys, os, json
import numpy as np, torch
args = sys.argv[1:]
sys.argv = [sys.argv[0], args[0], "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
ROT = {"w01_012", "w05_006", "w05_013"}
TF = {"id": lambda t: t, "rot180": lambda t: torch.flip(t, (0, 1)), "flipLR": lambda t: torch.flip(t, (1,)), "flipUD": lambda t: torch.flip(t, (0,))}
sums = {}
for p in people:
    num = torch.zeros(O.H, O.W, device=O.dev); den = torch.zeros_like(num); parts = {}
    for i in np.where(O.who == p)[0]:
        if p == "toorisugari" and str(O.names[i]) in ROT:
            continue
        a, b = O.contrib(i); num += a; den += b; parts[i] = (a, b)
    sums[p] = (num, den, parts)


def pce_t(i, K, tf):
    r = TF[tf](O.R[i].to(O.dev).float()); x = TF[tf](O.I[i].to(O.dev).float()); w = TF[tf](O.Wt[i].to(O.dev).float()).sqrt()
    a = r * w; b = x * K * w; a = a - a.mean(); b = b - b.mean()
    xc = torch.fft.irfft2(torch.fft.rfft2(a.cpu()) * torch.fft.rfft2(b.cpu()).conj(), s=(O.H, O.W))
    peak = xc[0, 0].item(); m = torch.ones_like(xc, dtype=torch.bool)
    for dy in range(-5, 6):
        m[dy % O.H, :6] = False; m[dy % O.H, O.W - 5:] = False
    return float(np.sign(peak) * peak * peak / (xc[m] ** 2).mean().item())


out = []
for i in range(O.N):
    row = {"who": str(O.who[i]), "name": str(O.names[i])}
    for p in people:
        num, den, parts = sums[p]
        if i in parts:
            a, b = parts[i]; K = O.zm((num - a) / (den - b + 1.0))
        else:
            K = O.zm(num / (den + 1.0))
        for tf in TF:
            row[f"{p}|{tf}"] = pce_t(i, K, tf)
    best = max((k for k in row if "|" in k), key=lambda k: row[k])
    row["best"] = best; row["best_pce"] = row[best]
    out.append(row)
    print(f"{row['who'][:5]:5s} {row['name']} best={best:22s} {row[best]:7.1f}", flush=True)
json.dump(out, open(args[1], "w"), indent=0)
