"""Orientation-aware PRNU attribution on cleaned residuals.

    python v2/attrib.py work out.json [--rounds 3] [--thr 4]
1. per credited camera: EM over orientation (id / rot180) -> MLE fingerprint K_c
2. LOO accuracy on credited photos
3. anonymous: correlation in both orientations vs every K_c; transductive rounds add
   confident anonymous photos (z-margin >= thr) to the fingerprints.
Score = NCC(T_o(W_i), T_o(I_i) * K_c) over the photo's mask, reported as z = score * sqrt(N_i).
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("work"); ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--thr", type=float, default=4.0)
args = ap.parse_args()
meta = json.load(open(f"{args.work}/meta.json")); names = meta["names"]
W = np.load(f"{args.work}/W.npy", mmap_mode="r"); I = np.load(f"{args.work}/I.npy", mmap_mode="r")
n = len(names); who = np.array([s.split("__")[0] for s in names])
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
ORI = ("id", "rot")
T = lambda a, o: a if o == "id" else a[::-1, ::-1]

w32 = [None] * n; i32 = [None] * n
def get(i):
    if w32[i] is None:
        w32[i] = np.ascontiguousarray(W[i], dtype=np.float32); i32[i] = np.ascontiguousarray(I[i], dtype=np.float32)
    return w32[i], i32[i]

def contrib(i, o):
    w, x = get(i)
    return T(w * x, o), T(x * x, o)

class FP:
    def __init__(self):
        self.num = 0; self.den = 0; self.members = {}
    def add(self, i, o):
        a, b = contrib(i, o); self.num = self.num + a; self.den = self.den + b; self.members[i] = o
    def K(self, exclude=None):
        num, den = self.num, self.den
        if exclude is not None and exclude in self.members:
            a, b = contrib(exclude, self.members[exclude]); num = num - a; den = den - b
        K = num / (den + 1.0)
        return K - K.mean()

def score(i, K, o):
    w, x = get(i)
    w = T(w, o); y = T(x, o) * K
    m = T(x, o) > 0
    a = w[m]; b = y[m]; a = a - a.mean(); b = b - b.mean()
    c = float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-12))
    return c * np.sqrt(m.sum())

# 1) credited orientation EM
fps = {}; orient = {}
for p in people:
    idx = list(np.where(who == p)[0]); o = {i: "id" for i in idx}
    for it in range(5):
        fp = FP()
        for i in idx: fp.add(i, o[i])
        new = {}
        for i in idx:
            K = fp.K(exclude=i)
            new[i] = max(ORI, key=lambda oo: score(i, K, oo))
        changed = sum(new[i] != o[i] for i in idx); o = new
        if changed == 0: break
    fp = FP()
    for i in idx: fp.add(i, o[i])
    fps[p] = fp; orient.update(o)
    print(p, "rot count", sum(v == "rot" for v in o.values()), "of", len(idx), flush=True)

def all_scores(i, exclude_self=True):
    res = {}
    for p in people:
        K = fps[p].K(exclude=i if exclude_self else None)
        for o in ORI:
            res[f"{p}|{o}"] = score(i, K, o)
    return res

# 2) LOO on credited
ok = 0; rows = []
for i in np.where(who != "anonymous")[0]:
    s = all_scores(i)
    best = max(s, key=s.get); pred = best.split("|")[0]
    ok += pred == who[i]
    rows.append({"name": names[i], "true": who[i], "pred": pred, **s})
print("credited LOO acc %.3f" % (ok / len(rows)), flush=True)
for p in people:
    sub = [r for r in rows if r["true"] == p]
    print("  ", p, sum(r["pred"] == p for r in sub), "/", len(sub))

# 3) anonymous with transductive rounds
anon = list(np.where(who == "anonymous")[0])
assigned = {}
for rnd in range(args.rounds + 1):
    out = []
    for i in anon:
        s = all_scores(i)
        per = {p: max(s[f"{p}|id"], s[f"{p}|rot"]) for p in people}
        ranked = sorted(per, key=per.get, reverse=True)
        best, second = ranked[0], ranked[1]
        o = "id" if s[f"{best}|id"] >= s[f"{best}|rot"] else "rot"
        out.append({"name": names[i], "pred": best, "ori": o, "z": per[best], "margin": per[best] - per[second], "second": second, **s})
    conf = [r for r in out if r["z"] >= args.thr and r["margin"] >= args.thr / 2]
    print(f"round {rnd}: confident {len(conf)}/{len(out)}", {p: sum(r['pred'] == p for r in conf) for p in people}, flush=True)
    if rnd == args.rounds: break
    # rebuild fingerprints: credited + confident anonymous
    for p in people:
        for r in conf:
            i = names.index(r["name"])
            if r["pred"] == p and i not in fps[p].members:
                fps[p].add(i, r["ori"])
json.dump({"credited": rows, "anonymous": out, "orient": {names[i]: o for i, o in orient.items()}}, open(args.out, "w"), indent=1)
import collections
print("final anon counts", collections.Counter(r["pred"] for r in out))
