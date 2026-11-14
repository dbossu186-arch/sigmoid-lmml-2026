"""Score every credited photo against the anonymous-based fingerprint of each person (independent photos).
Reveals credited photos that do not come from the same camera as the rest of that person's photos.
usage: cred_vs_anon.py <cache> <attrib.json> <minbest>"""
import sys, os, json
import numpy as np, torch
args = sys.argv[1:]
sys.argv = [sys.argv[0], args[0], args[1]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
att = json.load(open(args[1]))["anon"]; minb = float(args[2])
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
name2i = {(O.who[i], O.names[i]): i for i in range(O.N)}
Ks = {}
for p in people:
    idx = [name2i[("anonymous", r["name"])] for r in att if r["pred"] == p and r["best"] >= minb]
    num = torch.zeros(O.H, O.W, device=O.dev); den = torch.zeros_like(num)
    for i in idx:
        a, b = O.contrib(i); num += a; den += b
    Ks[p] = O.zm(num / (den + 1.0)); print(p, "anon n", len(idx), flush=True)
for p in people:
    for i in np.where(O.who == p)[0]:
        s = {q: O.pce(i, Ks[q]) for q in people}
        best = max(s, key=s.get)
        print(f"CRED {p:12s} {O.names[i]} best={best:12s} " + " ".join(f"{q[:4]}={s[q]:6.1f}" for q in people), flush=True)
