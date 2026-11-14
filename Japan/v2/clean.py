"""Clean residuals: DFT Wiener whitening (kills JPEG grid / periodic peaks), then remove the
component common to every photo (effect / pipeline pattern, in image coordinates).

    python v2/clean.py work/feats work/clean.npy work/meta.json
Saves X: float16 [n, H*W] with x_i = (w_i - I_i * Kc) * I_i, zero-mean, unit norm,
where w_i is the whitened residual and Kc the common multiplicative pattern.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import prnu_lib as P  # noqa: E402


def whiten(f):
    z = np.load(f)
    r = z["r"].astype(np.float32); I = z["I"].astype(np.float32); m = z["m"].astype(np.float32)
    w = P.wiener_dft(r) * m
    w = P.zero_mean_total(w) * m
    return w.astype(np.float16), I.astype(np.float16), m.astype(bool)


def main():
    feats = sorted(Path(sys.argv[1]).glob("*.npz"))
    with Pool(12) as pool:
        out = pool.map(whiten, feats)
    H, W = out[0][0].shape
    n = len(out)
    num = np.zeros((H, W), np.float64); den = np.zeros((H, W), np.float64)
    for w, I, m in out:
        I = I.astype(np.float32); num += w.astype(np.float32) * I; den += I * I
    np.save(Path(sys.argv[2]).with_name("common_Kc.npy"), (num / (den + 1)).astype(np.float32))
    X = np.empty((n, H * W), np.float16)
    Ws = np.empty((n, H, W), np.float16); Is = np.empty((n, H, W), np.float16)
    betas = []
    for k, (w, I, m) in enumerate(out):
        w = w.astype(np.float32); I = I.astype(np.float32); m = m.astype(np.float32)
        # leave-one-out common pattern, fitted per photo (no self-subtraction bias)
        Kc = ((num - w * I) / (den - I * I + 1)).astype(np.float32)
        p = I * Kc * m
        beta = float((w * m * p).sum() / ((p * p).sum() + 1e-9)); betas.append(beta)
        wc = (w - beta * p) * m
        wc = P.zero_mean_total(wc) * m
        x = wc * I
        x -= x.mean(); x /= np.linalg.norm(x) + 1e-9
        X[k] = x.ravel()
        Ws[k] = wc; Is[k] = I
    print("beta quantiles", np.round(np.quantile(betas, [0, .25, .5, .75, 1]), 3))
    np.save(sys.argv[2], X)
    np.save(Path(sys.argv[2]).with_name("W.npy"), Ws)
    np.save(Path(sys.argv[2]).with_name("I.npy"), Is)
    json.dump({"names": [f.stem for f in feats], "H": H, "W": W}, open(sys.argv[3], "w"))
    print("saved", X.shape)


if __name__ == "__main__":
    main()
