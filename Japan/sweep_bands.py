"""Which spatial-frequency band / channel carries the camera signal? LOO NCC on credited."""
import sys, glob, os
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P

BANDS = [("hp0.5", 0, 0.5), ("hp1", 0, 1), ("dog1-2", 1, 2), ("dog2-4", 2, 4), ("dog4-8", 4, 8), ("dog8-16", 8, 16),
         ("dog16-32", 16, 32), ("lp32", 32, None)]


def band(x, a, b):
    lo = ndimage.gaussian_filter(x, a) if a else x
    if b is None:
        return lo
    return lo - ndimage.gaussian_filter(x, b)


def work(p):
    rgb = P.load_rgb(p)
    m = P.caption_mask(rgb)
    out = {}
    for ci, cname in enumerate("RGBY"):
        x = rgb[..., ci] if ci < 3 else rgb.mean(2)
        for bn, a, b in BANDS:
            r = band(x, a, b)
            if bn == "lp32":
                r = r / (ndimage.gaussian_filter(x, 64) + 1) - 1  # relative low-frequency shading
            r = P.zero_mean_total(r * m) * m
            out[(cname, bn)] = r[240:720].astype(np.float16)
    return p, out


def main():
    data = sys.argv[1]
    paths = sorted(glob.glob(f"{data}/credited/*/*.jpg"))
    owners = np.array([p.split("/")[-2] for p in paths])
    with Pool(8) as pool:
        res = dict(pool.imap(work, paths, chunksize=2))
    keys = res[paths[0]].keys()
    people = sorted(set(owners))
    for k in keys:
        X = np.stack([res[p][k].ravel().astype(np.float32) for p in paths])
        X -= X.mean(1, keepdims=True)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        G = X @ X.T
        np.fill_diagonal(G, np.nan)
        # LOO: score of query vs person = mean NCC to that person's other images (approx fingerprint corr)
        S = np.stack([np.nanmean(G[:, owners == q], axis=1) for q in people], 1)
        pred = np.array(people)[S.argmax(1)]
        acc = (pred == owners).mean()
        same = np.nanmean([np.nanmean(G[i, owners == owners[i]]) for i in range(len(paths))])
        diff = np.nanmean([np.nanmean(G[i, owners != owners[i]]) for i in range(len(paths))])
        ty = owners == "toorisugari"; yk = owners == "yakin"
        ty_yk = np.nanmean(G[np.ix_(ty, yk)])
        print(f"{k[0]} {k[1]:9s} acc={acc:.3f} same={same:.5f} diff={diff:.5f} ratio={same/(abs(diff)+1e-9):6.2f} tooriXyakin={ty_yk:.5f}", flush=True)


if __name__ == "__main__":
    main()
