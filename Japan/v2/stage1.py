"""Per-photo PRNU features: wavelet noise residual (mean of R,G,B residuals), luminance, caption mask.

    python v2/stage1.py data/dataset work/feats
Writes <who>__<stem>.npz with r (residual), I (luminance), m (mask), all float16 / bool.
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import prnu_lib as P  # noqa: E402


def one(args):
    path, out = args
    if out.exists():
        return
    rgb = P.load_rgb(path)
    m = P.caption_mask(rgb)
    chans = np.stack([P.noise_extract_channel(rgb[..., c]) for c in range(3)])
    r = P.zero_mean_total(chans.mean(0) * m) * m
    I = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) * m
    np.savez(out, r=r.astype(np.float16), I=I.astype(np.float16), m=m.astype(bool),
             rc=(chans * m).astype(np.float16))


def main():
    data, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, outdir / f"{p.parent.name}__{p.stem}.npz")
            for p in sorted(data.glob("credited/*/*.jpg")) + sorted(data.glob("anonymous/*.jpg"))]
    with Pool(int(os.environ.get("NPROC", "12"))) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, jobs)):
            if i % 50 == 0:
                print(i, flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    main()
