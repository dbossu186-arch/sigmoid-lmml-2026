"""Pairwise residual correlations, identity and 180-degree rotation.

    python v2/gram.py work/feats work/gram.npz
x_i = r_i * I_i (PRNU signal is proportional to luminance), zero-mean, unit norm.
G_id[i,j] = <x_i, x_j>,  G_rot[i,j] = <x_i, rot180(x_j)>.
"""
import sys
from pathlib import Path
import numpy as np

feats = sorted(Path(sys.argv[1]).glob("*.npz"))
names = [f.stem for f in feats]
X = None
for k, f in enumerate(feats):
    z = np.load(f)
    x = z["r"].astype(np.float32) * z["I"].astype(np.float32)
    x -= x.mean()
    x /= np.linalg.norm(x) + 1e-9
    if X is None:
        X = np.empty((len(feats), x.size), np.float32)
        shape = x.shape
    X[k] = x.ravel()
Xr = X.reshape(len(feats), *shape)[:, ::-1, ::-1].reshape(len(feats), -1).copy()
G_id = X @ X.T
G_rot = X @ Xr.T
np.savez(sys.argv[2], names=np.array(names), G_id=G_id, G_rot=G_rot)
print("saved", G_id.shape)
