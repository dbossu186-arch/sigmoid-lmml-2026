"""Whitened residuals with no common-pattern subtraction; saves W_white.npy and gram.

    python v2/white_only.py work/feats work
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import prnu_lib as P  # noqa: E402

def whiten(f):
    z = np.load(f)
    r = z["r"].astype(np.float32); m = z["m"].astype(np.float32)
    w = P.wiener_dft(r) * m
    return (P.zero_mean_total(w) * m).astype(np.float16)

if __name__ == "__main__":
    feats = sorted(Path(sys.argv[1]).glob("*.npz")); work = Path(sys.argv[2])
    with Pool(12) as pool:
        Ws = np.stack(pool.map(whiten, feats))
    np.save(work / "W_white.npy", Ws)
    I = np.load(work / "I.npy", mmap_mode="r")
    n, H, Wd = Ws.shape
    X = np.empty((n, H * Wd), np.float32)
    for k in range(n):
        x = Ws[k].astype(np.float32) * I[k].astype(np.float32); x -= x.mean(); x /= np.linalg.norm(x) + 1e-9
        X[k] = x.ravel()
    Xr = X.reshape(n, H, Wd)[:, ::-1, ::-1].reshape(n, -1).copy()
    np.savez(work / "gram_white.npz", names=np.array([f.stem for f in feats]), G_id=X @ X.T, G_rot=X @ Xr.T)
    print("done")
