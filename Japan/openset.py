"""Open-set check: do the unattributed anonymous photos share an unknown camera?
For a set S, score each i in S by PCE against the fingerprint of S minus i; iterate keeping members."""
import sys, json, os, glob
import numpy as np, torch
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]] + sys.argv[3:]
cache, attjson = sys.argv[1], sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 and __name__ == "__main__" else 8.0
# reuse attribute.py helpers without running its main: minimal re-implementation
torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
FFTDEV = "cuda" if dev == "cuda" else "cpu"
files = sorted(glob.glob(f"{cache}/*.npz"))
who = np.array([os.path.basename(f).split("__")[0] for f in files])
names = np.array([os.path.basename(f).split("__")[1][:-4] for f in files])
R, I, Wt = [], [], []
for f in files:
    d = np.load(f); R.append(torch.from_numpy(d["r"])); I.append(torch.from_numpy(d["I"])); Wt.append(torch.from_numpy(d["w"]))
R = torch.stack(R); I = torch.stack(I); Wt = torch.stack(Wt)
N, H, W = R.shape


def zm(k):
    k = k - k.mean(-1, keepdim=True); return k - k.mean(-2, keepdim=True)


def contrib(i):
    r = R[i].to(dev).float(); x = I[i].to(dev).float(); w = Wt[i].to(dev).float()
    return r * x * w, x * x * w


def pce(i, K, excl=5):
    r = R[i].to(dev).float(); x = I[i].to(dev).float(); w = Wt[i].to(dev).float().sqrt()
    a = r * w; b = x * K * w; a = a - a.mean(); b = b - b.mean()
    xc = torch.fft.irfft2(torch.fft.rfft2(a.cpu()) * torch.fft.rfft2(b.cpu()).conj(), s=(H, W))
    peak = xc[0, 0].item()
    m = torch.ones_like(xc, dtype=torch.bool)
    for dy in range(-excl, excl + 1):
        m[dy % H, :excl + 1] = False; m[dy % H, W - excl:] = False
    return float(np.sign(peak) * peak * peak / (xc[m] ** 2).mean().item())


def loo_scores(S):
    num = torch.zeros(H, W, device=dev); den = torch.zeros(H, W, device=dev)
    parts = {}
    for i in S:
        a, b = contrib(i); num += a; den += b; parts[i] = (a.cpu(), b.cpu())
    out = {}
    for i in S:
        a, b = parts[i]
        out[i] = pce(i, zm((num - a.to(dev)) / (den - b.to(dev) + 1.0)))
    return out


if __name__ == "__main__":
    att = json.load(open(attjson))["anon"]
    idx = {n: i for i, n in enumerate(names) if who[i] == "anonymous"}
    low = [idx[r["name"]] for r in att if r["best"] < thr]
    print("low set size", len(low))
    sc = loo_scores(low)
    for i in sorted(sc, key=lambda k: -sc[k]):
        print("LOW", names[i], round(sc[i], 1))
    # control: same-size random set of confident photos of mixed owners
    hi = [idx[r["name"]] for r in att if r["best"] >= 20]
    sc2 = loo_scores(hi[:len(low)])
    print("control (mixed confident) median", np.median(list(sc2.values())), "max", max(sc2.values()))
