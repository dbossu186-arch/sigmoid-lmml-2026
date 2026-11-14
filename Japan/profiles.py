"""Per-photo column / row fixed-pattern profiles.
col[x] = mean over valid rows of (g - horizontal blur(g)) ; row[y] likewise with vertical blur.
Both additive (offset) and multiplicative (divided by local brightness) versions.
usage: profiles.py <dataset> <out.npz>"""
import sys, glob, os
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P

data, out = sys.argv[1:3]
paths = sorted(glob.glob(f"{data}/credited/*/*.jpg")) + sorted(glob.glob(f"{data}/anonymous/*.jpg"))
who = [p.split("/")[-2] for p in paths]; names = [os.path.basename(p)[:-4] for p in paths]
colA, colM, rowA, rowM, colRGB = [], [], [], [], []
for p in paths:
    rgb = P.load_rgb(p); m = P.caption_mask(rgb)
    g = rgb.mean(2)
    hx = g - ndimage.gaussian_filter1d(g, 2.0, axis=1)
    hy = g - ndimage.gaussian_filter1d(g, 2.0, axis=0)
    base = ndimage.gaussian_filter(g, 8) + 5
    # robust-ish: down-weight strong edges
    wx = m / (1 + (ndimage.gaussian_filter(hx * hx, 3) / 16))
    wy = m / (1 + (ndimage.gaussian_filter(hy * hy, 3) / 16))
    colA.append((hx * wx).sum(0) / (wx.sum(0) + 1e-6)); colM.append(((hx / base) * wx).sum(0) / (wx.sum(0) + 1e-6))
    rowA.append((hy * wy).sum(1) / (wy.sum(1) + 1e-6)); rowM.append(((hy / base) * wy).sum(1) / (wy.sum(1) + 1e-6))
    c3 = []
    for c in range(3):
        h = rgb[..., c] - ndimage.gaussian_filter1d(rgb[..., c], 2.0, axis=1)
        c3.append((h * wx).sum(0) / (wx.sum(0) + 1e-6))
    colRGB.append(np.stack(c3))
np.savez(out, who=np.array(who), names=np.array(names), colA=np.array(colA), colM=np.array(colM),
         rowA=np.array(rowA), rowM=np.array(rowM), colRGB=np.array(colRGB))
print("done", len(paths))
