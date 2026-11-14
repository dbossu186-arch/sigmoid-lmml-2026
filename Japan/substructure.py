"""Is a person's group (credited + anonymous members, orientation-aligned) one camera or two?
- pairwise NCC of aligned weighted residuals, 2-way spectral split
- each member scored against credited-only fingerprints of every person (orientation-aligned)
usage: substructure.py <cache> <joint.json> <person>"""
import sys, os, json
import numpy as np, torch
args = sys.argv[1:]
sys.argv = [sys.argv[0], args[0], "x"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset as O
J = json.load(open(args[1])); person = args[2]
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
idx = {(O.who[i], str(O.names[i])): i for i in range(O.N)}
mem = {idx[(r["who"], r["name"])]: tuple(r["member"]) for r in J if r["member"]}


def T(t, o):
    return t if o == "id" else torch.flip(t, (0, 1))


def pce(i, K, o):
    r = T(O.R[i].to(O.dev).float(), o); x = T(O.I[i].to(O.dev).float(), o); w = T(O.Wt[i].to(O.dev).float(), o).sqrt()
    a = r * w; b = x * K * w; a = a - a.mean(); b = b - b.mean()
    xc = torch.fft.irfft2(torch.fft.rfft2(a.to(O.FFTDEV)) * torch.fft.rfft2(b.to(O.FFTDEV)).conj(), s=(O.H, O.W))
    peak = xc[0, 0].item(); m = torch.ones_like(xc, dtype=torch.bool)
    for dy in range(-5, 6):
        m[dy % O.H, :6] = False; m[dy % O.H, O.W - 5:] = False
    return float(np.sign(peak) * peak * peak / (xc[m] ** 2).mean().item())


credK = {}
for p in people:
    num = torch.zeros(O.H, O.W, device=O.dev); den = torch.zeros_like(num); parts = {}
    for i, (q, o) in mem.items():
        if O.who[i] == p:
            a, b = O.contrib(i); a, b = T(a, o), T(b, o); num += a; den += b; parts[i] = (a.half().cpu(), b.half().cpu())
    credK[p] = (num, den, parts)

grp = [i for i, (q, o) in mem.items() if q == person]
print(f"{person}: {len(grp)} members ({sum(O.who[i] != 'anonymous' for i in grp)} credited)")
feats = []
for i in grp:
    o = mem[i][1]
    a = T((O.R[i].float() * O.I[i].float() * O.Wt[i].float().sqrt()), o)
    feats.append(a.flatten())
X = torch.stack(feats); X = X - X.mean(0, keepdim=True); X = X - X.mean(1, keepdim=True); X = X / X.norm(dim=1, keepdim=True)
G = (X @ X.T).numpy(); np.fill_diagonal(G, 0)
off = G[np.triu_indices(len(grp), 1)]
print("pairwise mean %.5f sd %.5f" % (off.mean(), off.std()))
# 2-way spectral split on positive part
A = np.clip((G - off.mean()) / off.std(), 0, None)
D = np.diag(1 / np.sqrt(A.sum(1) + 1e-9)); Lsym = np.eye(len(grp)) - D @ A @ D
w, v = np.linalg.eigh(Lsym); f = v[:, 1]
lab = (f > np.median(f)).astype(int)
for c in (0, 1):
    m = lab == c
    within = G[np.ix_(m, m)][~np.eye(m.sum(), dtype=bool)].mean()
    print(f"  split {c}: n={m.sum()} within-mean {within:.5f}")
print("  between-mean %.5f" % G[np.ix_(lab == 0, lab == 1)].mean())
print("\nmember scores vs credited-only fingerprints (LOO for credited members):")
for k, i in enumerate(grp):
    o = mem[i][1]; s = {}
    for p in people:
        num, den, parts = credK[p]
        if i in parts:
            a, b = parts[i]; K = O.zm((num - a.to(O.dev).float()) / (den - b.to(O.dev).float() + 1.0))
        else:
            K = O.zm(num / (den + 1.0))
        s[p] = pce(i, K, o)
    print(f"  {O.who[i][:5]}/{O.names[i]} ori={o:6s} split={lab[k]}  " + " ".join(f"{p[:4]}={s[p]:6.1f}" for p in people), flush=True)
