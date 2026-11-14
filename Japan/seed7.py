"""Seed an 'unknown camera' fingerprint from a few photos and score every photo against it (LOO for members)."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], sys.argv[1], "x"] + sys.argv[2:]
import openset as O
seed = sys.argv[3].split(",")
idx = [int(np.where(O.names == n)[0][0]) for n in seed]
for rnd in range(3):
    num = torch.zeros(O.H, O.W, device=O.dev); den = torch.zeros_like(num); parts = {}
    for i in idx:
        a, b = O.contrib(i); num += a; den += b; parts[i] = (a, b)
    K = O.zm(num / (den + 1.0))
    sc = {}
    for i in range(O.N):
        if i in parts:
            a, b = parts[i]; sc[i] = O.pce(i, O.zm((num - a) / (den - b + 1.0)))
        else:
            sc[i] = O.pce(i, K)
    order = sorted(sc, key=lambda k: -sc[k])
    print(f"round {rnd} seed={len(idx)}")
    print("  top:", [(O.who[i][:5], O.names[i], round(sc[i], 1)) for i in order[:40]])
    cred = [sc[i] for i in range(O.N) if O.who[i] != "anonymous"]
    print("  credited max %.1f p99 %.1f" % (max(cred), np.percentile(cred, 99)))
    idx = [i for i in order if sc[i] > 15 and O.who[i] == "anonymous"]
