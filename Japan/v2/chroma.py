"""Gram matrices on chroma residuals (R-G and B-G of per-channel wavelet residuals).

    python v2/chroma.py work/feats work/gram_chroma.npz
"""
import sys
from pathlib import Path
import numpy as np

feats = sorted(Path(sys.argv[1]).glob("*.npz"))
names = [f.stem for f in feats]
out = {"names": np.array(names)}
for tag, (a, b) in {"rg": (0, 1), "bg": (2, 1)}.items():
    X = None
    for k, f in enumerate(feats):
        z = np.load(f)
        rc = z["rc"].astype(np.float32); m = z["m"].astype(np.float32); I = z["I"].astype(np.float32)
        x = (rc[a] - rc[b]) * m
        x = x - x.mean(axis=0, keepdims=True); x = x - x.mean(axis=1, keepdims=True); x *= m
        x = x * I
        x -= x.mean(); x /= np.linalg.norm(x) + 1e-9
        if X is None:
            X = np.empty((len(feats), x.size), np.float32); shape = x.shape
        X[k] = x.ravel()
    Xr = X.reshape(len(feats), *shape)[:, ::-1, ::-1].reshape(len(feats), -1).copy()
    out[f"{tag}_id"] = X @ X.T; out[f"{tag}_rot"] = X @ Xr.T
    del Xr
    print(tag, "done", flush=True)
np.savez(sys.argv[2], **out)
