"""PCE with full peak search (any circular shift), both orientations, every photo vs every joint fingerprint.
Flags photos whose best peak is NOT at zero shift (cropped / shifted / re-framed images).
usage: peaksearch.py <cache> <joint.json> <out.json>"""
import sys, os, json
import numpy as np, torch
args = sys.argv[1:]
sys.argv = [sys.argv[0], args[0], "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
J = json.load(open(args[1]))
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
idx = {(O.who[i], str(O.names[i])): i for i in range(O.N)}
mem = {idx[(r["who"], r["name"])]: tuple(r["member"]) for r in J if r["member"]}


def T(t, o):
    return t if o == "id" else torch.flip(t, (0, 1))


num = {p: torch.zeros(O.H, O.W, device=O.dev) for p in people}; den = {p: torch.zeros(O.H, O.W, device=O.dev) for p in people}
parts = {}
for i, (p, o) in mem.items():
    a, b = O.contrib(i); a, b = T(a, o), T(b, o); num[p] += a; den[p] += b; parts[i] = (a.half().cpu(), b.half().cpu())
Kf = {p: O.zm(num[p] / (den[p] + 1.0)) for p in people}


def peak_pce(i, K, o):
    r = T(O.R[i].to(O.dev).float(), o); x = T(O.I[i].to(O.dev).float(), o); w = T(O.Wt[i].to(O.dev).float(), o).sqrt()
    a = r * w; b = x * K * w; a = a - a.mean(); b = b - b.mean()
    xc = torch.fft.irfft2(torch.fft.rfft2(a.to(O.FFTDEV)) * torch.fft.rfft2(b.to(O.FFTDEV)).conj(), s=(O.H, O.W))
    k = int(torch.argmax(xc).item()); py, px = divmod(k, O.W)
    peak = xc[py, px].item()
    m = torch.ones_like(xc, dtype=torch.bool)
    ys = [(py + d) % O.H for d in range(-5, 6)]; xs = [(px + d) % O.W for d in range(-5, 6)]
    m[np.ix_(ys, xs)] = False
    e = (xc[m] ** 2).mean().item()
    z0 = xc[0, 0].item()
    sy = py if py < O.H // 2 else py - O.H; sx = px if px < O.W // 2 else px - O.W
    return peak * peak / e, (sy, sx), np.sign(z0) * z0 * z0 / e


out = []
for i in range(O.N):
    best = None
    for p in people:
        if i in mem and mem[i][0] == p:
            a, b = parts[i]; K = O.zm((num[p] - a.to(O.dev).float()) / (den[p] - b.to(O.dev).float() + 1.0))
        else:
            K = Kf[p]
        for o in ("id", "rot180"):
            pc, sh, z = peak_pce(i, K, o)
            if best is None or pc > best[0]:
                best = (pc, sh, z, p, o)
    pc, sh, z, p, o = best
    out.append(dict(who=str(O.who[i]), name=str(O.names[i]), pce=pc, shift=sh, pce0=z, cam=p, ori=o))
    flag = "  <-- SHIFTED" if sh != (0, 0) and pc > 20 else ""
    print(f"{O.who[i][:5]}/{O.names[i]} best {p}|{o} peakPCE {pc:7.1f} at {sh}  zeroPCE {z:7.1f}{flag}", flush=True)
json.dump(out, open(args[2], "w"), indent=0)
