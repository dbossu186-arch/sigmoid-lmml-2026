"""Per-person robust (median) residual map; look for sparse fixed-position outliers (hot/dead pixels)."""
import sys, glob, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P
from viz_fp import gblur, zm, dev

data, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)


def resid(p):
    rgb = P.load_rgb(p); m = P.caption_mask(rgb)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    r = (x - gblur(x, 1.0))[0].cpu().numpy()
    r[:, m == 0] = np.nan
    return r


people = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
for who in people:
    R = np.stack([resid(p) for p in sorted(glob.glob(f"{data}/credited/{who}/*.jpg"))])  # N,3,H,W
    med = np.nanmedian(R, axis=0)
    sgn = np.nanmean(np.sign(R), axis=0)
    np.save(f"{out}/med_{who}.npy", med.astype(np.float32)); np.save(f"{out}/sgn_{who}.npy", sgn.astype(np.float32))
    g = np.nan_to_num(med.mean(0))
    z = (g - np.median(g)) / (1.4826 * np.median(np.abs(g - np.median(g))) + 1e-9)
    idx = np.argsort(-np.abs(z).ravel())[:15]
    ys, xs = np.unravel_index(idx, z.shape)
    s = np.nan_to_num(sgn.mean(0))
    print(who, "robust sd", np.nanstd(g).round(3), "frac |sgn|>0.6:", (np.abs(s) > 0.6).mean().round(5))
    print("   top:", [(int(y), int(x), round(float(z[y, x]), 1)) for y, x in zip(ys, xs)])
