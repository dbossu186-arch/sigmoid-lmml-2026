"""Pairwise-gram LOO check on credited photos: score_c(i) = mean_j in c, j!=i of max(G_id, G_rot)."""
import json, sys
import numpy as np
X = np.load(sys.argv[1]).astype(np.float32)
meta = json.load(open(sys.argv[2])); H, W = meta["H"], meta["W"]; names = meta["names"]
n = len(names)
Xr = X.reshape(n, H, W)[:, ::-1, ::-1].reshape(n, -1)
Gi = X @ X.T; Gr = X @ Xr.T
np.savez(sys.argv[3], G_id=Gi, G_rot=Gr, names=np.array(names))
who = np.array([s.split("__")[0] for s in names])
people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
S = np.maximum(Gi, Gr); np.fill_diagonal(S, np.nan)
ok = 0; tot = 0
for i in np.where(who != "anonymous")[0]:
    sc = [np.nanmean(S[i, who == p]) for p in people]
    ok += people[int(np.argmax(sc))] == who[i]; tot += 1
print("pairwise LOO acc %.3f" % (ok / tot))
for p in people:
    a = who == p
    print(p.ljust(12), " ".join("%6.1f" % (1e4 * np.nanmean(S[np.ix_(a, who == q)])) for q in people))
