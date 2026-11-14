"""Stage 1: compute wavelet noise residuals for all photos (parallel). Saves residuals.npz-ish cache."""
import sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P


def work(args):
    path, out = args
    if os.path.exists(out):
        return out
    rgb = P.load_rgb(path)
    m = P.caption_mask(rgb)
    chans, gray = P.residual_rgb(rgb, m)
    inten = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) * m
    np.savez(out, chans=chans.astype(np.float16), gray=gray, inten=inten.astype(np.float32),
             mask=m.astype(np.uint8))
    return out


def main():
    data = Path(sys.argv[1]); cache = Path(sys.argv[2]); cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in sorted((data / "credited").glob("*/*.jpg")):
        jobs.append((str(p), str(cache / f"cred_{p.parent.name}_{p.stem}.npz")))
    for p in sorted((data / "anonymous").glob("*.jpg")):
        jobs.append((str(p), str(cache / f"anon_{p.stem}.npz")))
    with Pool(14) as pool:
        for i, o in enumerate(pool.imap_unordered(work, jobs)):
            if i % 20 == 0:
                print(i, o, flush=True)


if __name__ == "__main__":
    main()
