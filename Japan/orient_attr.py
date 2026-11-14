"""Orientation-aware PRNU attribution.
Phones in landscape can be held either way round -> sensor pattern rotated by 180 deg in some photos.
1) per credited person: EM over orientation (id / rot180) to build an aligned MLE fingerprint
2) LOO CV on credited (both orientations tried)
3) anonymous: PCE in both orientations vs every fingerprint; transductive rounds add confident photos
usage: orient_attr.py <cache> <out_dir> [--thr 15] [--rounds 3]"""
import sys, os, json, argparse
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("out")
ap.add_argument("--thr", type=float, default=15.0); ap.add_argument("--rounds", type=int, default=3)
args = ap.parse_args()
sys.argv = [sys.argv[0], args.cache, "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
os.makedirs(args.out, exist_ok=True)
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
ORI = ("id", "rot180")


def T(t, o):
    return t if o == "id" else torch.flip(t, (0, 1))


def contrib(i, o):
    a, b = O.contrib(i)
    return T(a, o), T(b, o)


def pce(i, K, o):
    r = T(O.R[i].to(O.dev).float(), o); x = T(O.I[i].to(O.dev).float(), o); w = T(O.Wt[i].to(O.dev).float(), o).sqrt()
    a = r * w; b = x * K * w; a = a - a.mean(); b = b - b.mean()
    xc = torch.fft.irfft2(torch.fft.rfft2(a.to(O.FFTDEV)) * torch.fft.rfft2(b.to(O.FFTDEV)).conj(), s=(O.H, O.W))
    peak = xc[0, 0].item(); m = torch.ones_like(xc, dtype=torch.bool)
    for dy in range(-5, 6):
        m[dy % O.H, :6] = False; m[dy % O.H, O.W - 5:] = False
    return float(np.sign(peak) * peak * peak / (xc[m] ** 2).mean().item())


class FP:
    def __init__(self):
        self.num = torch.zeros(O.H, O.W, device=O.dev); self.den = torch.zeros_like(self.num); self.parts = {}

    def add(self, i, o):
        a, b = contrib(i, o); self.num += a; self.den += b; self.parts[i] = (a.half().cpu(), b.half().cpu(), o)

    def K(self, exclude=None):
        if exclude is not None and exclude in self.parts:
            a, b, _ = self.parts[exclude]; a = a.to(O.dev).float(); b = b.to(O.dev).float()
            return O.zm((self.num - a) / (self.den - b + 1.0))
        return O.zm(self.num / (self.den + 1.0))


# ---------- 1) credited orientation EM
ori = {}
cred = {p: list(np.where(O.who == p)[0]) for p in people}
fps = {}
for p in people:
    o_assign = {i: "id" for i in cred[p]}
    for it in range(4):
        fp = FP()
        for i in cred[p]:
            fp.add(i, o_assign[i])
        new = {}
        for i in cred[p]:
            K = fp.K(exclude=i)
            s = {o: pce(i, K, o) for o in ORI}
            new[i] = max(s, key=s.get)
        changed = sum(new[i] != o_assign[i] for i in cred[p])
        o_assign = new
        if changed == 0:
            break
    fps[p] = FP()
    for i in cred[p]:
        fps[p].add(i, o_assign[i])
    ori.update(o_assign)
    print(f"{p:12s} credited rot180: {sorted(str(O.names[i]) for i in cred[p] if o_assign[i] == 'rot180')}", flush=True)

# ---------- 2) LOO CV
cv = []
for p in people:
    for i in cred[p]:
        row = {}
        for q in people:
            K = fps[q].K(exclude=i if q == p else None)
            row[q] = max(pce(i, K, o) for o in ORI)
        pred = max(row, key=row.get)
        cv.append(dict(name=str(O.names[i]), true=p, pred=pred, **row))
acc = float(np.mean([c["true"] == c["pred"] for c in cv]))
print(f"LOO accuracy (orientation-aware) {acc:.3f}")
for p in people:
    rr = [c for c in cv if c["true"] == p]
    print(f"   {p:12s} {sum(c['pred'] == p for c in rr)}/{len(rr)}  wrong->{[c['pred'] for c in rr if c['pred'] != p]}", flush=True)

# ---------- 3) anonymous, transductive
anon = list(np.where(O.who == "anonymous")[0])
assigned = {}
for rnd in range(args.rounds + 1):
    cur = {p: FP() for p in people}
    for p in people:
        for i in cred[p]:
            cur[p].add(i, ori[i])
    for i, (p, o) in assigned.items():
        cur[p].add(i, o)
    res = []
    for i in anon:
        sc = {}
        for p in people:
            K = cur[p].K(exclude=i)
            for o in ORI:
                sc[f"{p}|{o}"] = pce(i, K, o)
        best = max(sc, key=sc.get)
        per_person = {p: max(sc[f"{p}|{o}"] for o in ORI) for p in people}
        srt = sorted(per_person.values(), reverse=True)
        res.append(dict(name=str(O.names[i]), pred=best.split("|")[0], ori=best.split("|")[1], best=srt[0], second=srt[1], **sc))
    new = {i: (r["pred"], r["ori"]) for i, r in zip(anon, res) if r["best"] >= args.thr and r["best"] >= 3 * max(r["second"], 1)}
    cnt = {p: sum(v[0] == p for v in new.values()) for p in people}
    print(f"round {rnd}: confident {len(new)}/{len(anon)} {cnt}", flush=True)
    if new == assigned:
        break
    assigned = new
json.dump(dict(cv=cv, acc=acc, anon=res, cred_ori={str(O.who[i]) + "/" + str(O.names[i]): ori[i] for i in ori}),
          open(f"{args.out}/orient_attrib.json", "w"), indent=1)
for p in people:
    np.save(f"{args.out}/K_{p}.npy", cur[p].K().cpu().numpy().astype(np.float32))
low = [r for r in res if r["best"] < args.thr]
print("low-confidence:", len(low), sorted(r["name"] for r in low))
