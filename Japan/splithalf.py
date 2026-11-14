"""Does a person's anonymous group share an extra pattern beyond their credited fingerprint?
Compare cos(K_anonA, K_anonB) with cos(K_anon*, K_cred) for each person.
usage: splithalf.py <cache> <attrib.json> <minbest>"""
import sys, glob, os, json
import numpy as np
from scipy import ndimage
cache, attj = sys.argv[1:3]; minb = float(sys.argv[3])
att = json.load(open(attj))["anon"]
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]

allf = sorted(glob.glob(f"{cache}/*.npz"))
M = None
for f in allf:
    d = np.load(f); a = d["r"].astype(np.float32) * d["I"].astype(np.float32)
    M = a if M is None else M + a
Mb = M / len(allf)


def K(files):
    num = 0; den = 0
    for f in files:
        d = np.load(f); r = d["r"].astype(np.float32); I = d["I"].astype(np.float32); w = d["w"].astype(np.float32)
        num = num + (r * I - Mb) * w; den = den + I * I * w
    k = num / (den + 1.0)
    k = k - k.mean(0, keepdims=True); k = k - k.mean(1, keepdims=True)
    return k


def cos(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


rng = np.random.default_rng(0)
for p in people:
    an = [f"{cache}/anonymous__{r['name']}.npz" for r in att if r["pred"] == p and r["best"] >= minb]
    cr = sorted(glob.glob(f"{cache}/{p}__*.npz"))
    if len(an) < 6:
        print(p, "too few anon", len(an)); continue
    vals = []
    for rep in range(3):
        perm = rng.permutation(len(an)); A = [an[i] for i in perm[::2]]; B = [an[i] for i in perm[1::2]]
        crp = rng.permutation(len(cr)); C1 = [cr[i] for i in crp[:len(cr) // 2]]; C2 = [cr[i] for i in crp[len(cr) // 2:]]
        kA, kB, kC1, kC2 = K(A), K(B), K(C1), K(C2)
        vals.append((cos(kA, kB), cos(kA, kC1), cos(kB, kC2), cos(kC1, kC2), len(A), len(C1)))
    v = np.array(vals)
    print(f"{p:12s} anon n={len(an):2d} | anonA-anonB {v[:,0].mean():.4f} | anon-cred {v[:,1:3].mean():.4f} | credA-credB {v[:,3].mean():.4f}", flush=True)
