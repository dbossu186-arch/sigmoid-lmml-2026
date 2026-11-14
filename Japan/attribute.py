"""MLE fingerprints + PCE attribution from a feature cache (see cache_feats.py).
usage: attribute.py <cache_dir> <out_dir> [--iter N]"""
import sys, glob, os, json, argparse
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("cache"); ap.add_argument("out")
ap.add_argument("--iter", type=int, default=0, help="transductive rounds adding confident anon photos")
ap.add_argument("--wpow", type=float, default=1.0)
ap.add_argument("--pce_thr", type=float, default=40.0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.set_num_threads(4)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

files = sorted(glob.glob(f"{args.cache}/*.npz"))
who = np.array([os.path.basename(f).split("__")[0] for f in files])
names = np.array([os.path.basename(f).split("__")[1][:-4] for f in files])
people = sorted(set(who) - {"anonymous"})
R, I, Wt = [], [], []
for f in files:
    d = np.load(f)
    R.append(torch.from_numpy(d["r"])); I.append(torch.from_numpy(d["I"])); Wt.append(torch.from_numpy(d["w"]))
R = torch.stack(R); I = torch.stack(I); Wt = torch.stack(Wt) ** args.wpow
N, H, W = R.shape
print("loaded", R.shape, dev, flush=True)


def zm(k):
    k = k - k.mean(-1, keepdim=True); return k - k.mean(-2, keepdim=True)


def contrib(i):
    r = R[i].to(dev).float(); x = I[i].to(dev).float(); w = Wt[i].to(dev).float()
    return r * x * w, x * x * w


def build(idx):
    num = torch.zeros(H, W, device=dev); den = torch.zeros(H, W, device=dev)
    for i in idx:
        a, b = contrib(i); num += a; den += b
    return num, den


def K_from(num, den):
    return zm(num / (den + 1.0))


def pce(i, K, excl=5):
    r = R[i].to(dev).float(); x = I[i].to(dev).float(); w = Wt[i].to(dev).float().sqrt()
    a = r * w; b = x * K * w
    a = a - a.mean(); b = b - b.mean()
    A = torch.fft.rfft2(a.cpu()); B = torch.fft.rfft2(b.cpu())
    xc = torch.fft.irfft2(A * B.conj(), s=(H, W))
    peak = xc[0, 0].item()
    xc2 = xc.clone()
    for dy in range(-excl, excl + 1):
        xc2[dy % H, :excl + 1] = 0; xc2[dy % H, -excl:] = 0
    m = torch.ones_like(xc, dtype=torch.bool)
    for dy in range(-excl, excl + 1):
        m[dy % H, :excl + 1] = False; m[dy % H, W - excl:] = False
    energy = (xc[m] ** 2).mean().item()
    return float(np.sign(peak) * peak * peak / energy)


# ---------------- credited LOO CV
cred_idx = {p: np.where(who == p)[0] for p in people}
sums = {p: build(cred_idx[p]) for p in people}
cv = []
for p in people:
    for i in cred_idx[p]:
        row = {}
        for q in people:
            num, den = sums[q]
            if q == p:
                a, b = contrib(i); num = num - a; den = den - b
            row[q] = pce(i, K_from(num, den))
        pred = max(row, key=row.get)
        cv.append(dict(name=str(names[i]), true=p, pred=pred, **row))
acc = np.mean([c["true"] == c["pred"] for c in cv])
print("LOO accuracy %.3f" % acc)
for p in people:
    rows = [c for c in cv if c["true"] == p]
    print("  %-12s correct %2d/%d  median own PCE %.1f  wrong->%s" % (p, sum(c["pred"] == p for c in rows), len(rows),
          np.median([c[p] for c in rows]), [c["pred"] for c in rows if c["pred"] != p]))


# ---------------- anonymous
anon = np.where(who == "anonymous")[0]
assign = {}
for it in range(args.iter + 1):
    fps = {}
    for p in people:
        num, den = sums[p]
        num = num.clone(); den = den.clone()
        for i, q in assign.items():
            if q == p:
                a, b = contrib(i); num += a; den += b
        fps[p] = (num, den)
    res = []
    for i in anon:
        row = {}
        for p in people:
            num, den = fps[p]
            if assign.get(i) == p:  # leave self out
                a, b = contrib(i); num = num - a; den = den - b
            row[p] = pce(i, K_from(num, den))
        srt = sorted(row.values(), reverse=True)
        pred = max(row, key=row.get)
        res.append(dict(name=str(names[i]), pred=pred, best=srt[0], second=srt[1], **row))
    new = {i: r["pred"] for i, r in zip(anon, res) if r["best"] > args.pce_thr}
    print(f"round {it}: confident {len(new)}/{len(anon)}", {p: sum(v == p for v in new.values()) for p in people}, flush=True)
    assign = new
json.dump(dict(cv=cv, acc=acc, anon=res), open(f"{args.out}/attrib.json", "w"), indent=1)
for r in res:
    print("ANON %s pred=%-11s best=%7.1f second=%7.1f | %s" % (r["name"], r["pred"], r["best"], r["second"],
          " ".join(f"{p[:4]}={r[p]:.0f}" for p in people)))
