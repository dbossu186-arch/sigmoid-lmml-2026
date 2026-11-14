"""Stage 2: MLE fingerprints, LOO CV on credited, PCE attribution of anonymous, pairwise anon correlations."""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P

cache = sys.argv[1]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(f"{cache}/*.npz"))
names, owners, W, I = [], [], [], []
for f in files:
    b = os.path.basename(f)[:-4]
    d = np.load(f)
    if b.startswith("cred_"):
        _, who, stem = b.split("_", 2)
    else:
        who, stem = "anon", b[5:]
    names.append(stem); owners.append(who)
    W.append(d["gray"].astype(np.float32)); I.append(d["inten"].astype(np.float32))
W = np.stack(W); I = np.stack(I)
owners = np.array(owners); names = np.array(names)
people = sorted(set(owners) - {"anon"})
print("loaded", W.shape, flush=True)


def fingerprint(idx):
    num = (W[idx] * I[idx]).sum(0); den = (I[idx] ** 2).sum(0) + 1.0
    K = num / den
    K = P.zero_mean_total(K)
    return P.wiener_dft(K)


def score(q, K):
    return P.pce(W[q], I[q] * K)


# ---- LOO CV
cred = np.where(owners != "anon")[0]
K_full = {p: fingerprint(np.where(owners == p)[0]) for p in people}
cv = []
for q in cred:
    row = {}
    for p in people:
        idx = np.where(owners == p)[0]
        idx = idx[idx != q]
        K = fingerprint(idx) if owners[q] == p else K_full[p]
        row[p] = score(q, K)
    pred = max(row, key=row.get)
    cv.append(dict(name=names[q], true=owners[q], pred=pred, **row))
    print("CV", names[q], owners[q], pred, {k: round(v) for k, v in row.items()}, flush=True)
acc = np.mean([c["true"] == c["pred"] for c in cv])
print("LOO accuracy", acc)

# ---- anonymous
anon = np.where(owners == "anon")[0]
res = []
for q in anon:
    row = {p: score(q, K_full[p]) for p in people}
    pred = max(row, key=row.get)
    res.append(dict(name=names[q], pred=pred, **row))
    print("ANON", names[q], pred, {k: round(v) for k, v in row.items()}, flush=True)

# ---- pairwise NCC between all residuals (zero shift)
Wf = (W * I).reshape(len(W), -1)  # weight by intensity (PRNU is multiplicative)
Wf = Wf - Wf.mean(1, keepdims=True)
Wf /= np.linalg.norm(Wf, axis=1, keepdims=True)
C = Wf @ Wf.T
np.save(f"{out}/pair_ncc.npy", C)
json.dump(dict(cv=cv, anon=res, names=names.tolist(), owners=owners.tolist(), acc=acc),
          open(f"{out}/stage2.json", "w"))
for p in people:
    np.save(f"{out}/K_{p}.npy", K_full[p].astype(np.float32))
