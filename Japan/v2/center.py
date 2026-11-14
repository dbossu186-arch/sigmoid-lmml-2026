"""Double-center the off-diagonal of G_id and G_rot (removes additive per-photo biases f_i + f_j).

    python v2/center.py work/gram_clean.npz work/gram_cc.npz
"""
import sys
import numpy as np
z = np.load(sys.argv[1]); out = {"names": z["names"]}
for key in ("G_id", "G_rot"):
    G = z[key].astype(np.float64).copy(); n = len(G)
    d = np.diag(G).copy(); np.fill_diagonal(G, np.nan)
    for _ in range(20):  # iterate: off-diagonal row/col means -> 0
        r = np.nanmean(G, axis=1); G = G - r[:, None] / 2 - r[None, :] / 2
    print(key, "residual row-mean max %.1e" % np.abs(np.nanmean(G, axis=1)).max())
    G[np.isnan(G)] = 0; np.fill_diagonal(G, d)
    out[key] = G
np.savez(sys.argv[2], **out)
