"""Per-photo PRNU features (Mac/MPS friendly). For each photo store, in float16:
   r  : noise residual (channel-averaged), zero-mean rows/cols, masked
   I  : luminance, masked
   w  : texture weight 1/(local residual energy + c)
usage: cache_feats.py <dataset> <cache_dir> <variant: hp0.5 | wav>"""
import sys, glob, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P
from viz_fp import gblur, zm, dev


def residual(rgb, variant):
    if variant == "wav":
        r = np.stack([P.noise_extract_channel(rgb[..., c]) for c in range(3)])
        return torch.from_numpy(r)[None].to(dev)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    s = float(variant[2:])
    return x - gblur(x, s)


def save_chroma(rgb, variant, out):
    """Grain is monochrome (identical in R,G,B) -> chroma residual is grain-free. JPEG is 4:2:0 so work at half res."""
    m = P.caption_mask(rgb)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    if variant == "cb":
        C = -0.168736 * R - 0.331264 * G + 0.5 * B
    elif variant == "cr":
        C = 0.5 * R - 0.418688 * G - 0.081312 * B
    elif variant == "rg":
        C = R - G
    else:
        C = B - G
    def half(a):
        return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))
    C2, Y2 = half(C), half(Y)
    m2 = (half(m) > 0.99).astype(np.float32)
    x = torch.from_numpy(C2)[None, None].to(dev)
    r = x - gblur(x, 0.7)
    mm = torch.from_numpy(m2)[None, None].to(dev)
    r = zm(r * mm) * mm
    w = 1.0 / (gblur(r * r, 3) + 0.05)
    np.savez(out, r=r[0, 0].cpu().numpy().astype(np.float16), I=(Y2 * m2).astype(np.float16),
             w=(w[0, 0].cpu().numpy() / 20.0).astype(np.float16))


def main():
    data, cache, variant = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(cache, exist_ok=True)
    paths = sorted(glob.glob(f"{data}/credited/*/*.jpg")) + sorted(glob.glob(f"{data}/anonymous/*.jpg"))
    for i, p in enumerate(paths):
        who = p.split("/")[-2]; stem = os.path.basename(p)[:-4]
        out = f"{cache}/{who}__{stem}.npz"
        if os.path.exists(out):
            continue
        rgb = P.load_rgb(p)
        if variant in ("cb", "cr", "rg", "bg"):
            save_chroma(rgb, variant, out)
            continue
        m = torch.from_numpy(P.caption_mask(rgb)).to(dev)[None, None]
        r = residual(rgb, variant)
        r = zm(r * m) * m
        rc = r.mean(1, keepdim=True)                                  # channel average
        rc = zm(rc) * m
        e = gblur(rc * rc, 4)
        w = 1.0 / (e + 1.0)
        x = torch.from_numpy(rgb).to(dev)
        I = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[None, None] * m
        np.savez(out, r=rc[0, 0].cpu().numpy().astype(np.float16), I=I[0, 0].cpu().numpy().astype(np.float16),
                 w=w[0, 0].cpu().numpy().astype(np.float16))
        if i % 25 == 0:
            print(i, out, flush=True)


if __name__ == "__main__":
    main()
