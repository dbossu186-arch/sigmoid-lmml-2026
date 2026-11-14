"""PRNU utilities: wavelet (Mihcak) denoiser residuals, MLE fingerprints, PCE.

Standard pipeline (Lukas/Fridrich/Goljan; Chen et al. 2008):
  W = I - F(I)  with F = wavelet-domain locally adaptive Wiener denoiser (db8, 4 levels)
  zero-mean rows/cols per channel, RGB -> gray combination, Wiener filtering in DFT domain
  K = sum(W_i * I_i) / sum(I_i^2)
  PCE of circular cross-correlation between W_query and I_query * K
"""
import numpy as np
import pywt
from PIL import Image
from scipy import ndimage

SIGMA = float(__import__("os").environ.get("PRNU_SIGMA", "3.0"))  # wavelet-denoiser noise std (grain ~4)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def caption_mask(rgb, thr=10.0):
    """1 where the pixel carries scene content; 0 on the caption (black bar or translucent box + text)
    and on saturated / clipped pixels."""
    gray = rgb.mean(axis=2)
    H, W = gray.shape
    m = np.ones(gray.shape, bool)
    rowdark = (gray < thr).mean(axis=1) > 0.6           # rows of an opaque black bar
    m[rowdark, :] = False
    # caption text: thin near-white unsaturated strokes on a dark surround
    white = (rgb.min(axis=2) > 190) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 35)
    surround = ndimage.uniform_filter(gray, 9)
    txt = white & (surround < 120)
    band = np.zeros(H, bool)
    band[:330] = True; band[H - 330:] = True
    txt &= band[:, None]
    rows = ndimage.uniform_filter1d(txt.sum(axis=1).astype(float), 5)
    trow = rows > 12
    if trow.any():
        lab, n = ndimage.label(ndimage.binary_closing(trow, iterations=20))
        for k in range(1, n + 1):
            rr = np.where(lab == k)[0]
            if len(rr) < 12:
                continue
            colcnt = txt[rr].sum(axis=0)
            cols = np.where(ndimage.uniform_filter1d(colcnt.astype(float), 31) > 0.2)[0]
            if len(cols) == 0:
                continue
            r0, r1 = max(0, rr.min() - 25), min(H, rr.max() + 26)
            c0, c1 = max(0, cols.min() - 40), min(W, cols.max() + 41)
            # the box background must be dark (caption plate), otherwise it is scene content
            sub = gray[r0:r1, c0:c1]
            if np.median(sub) > 70:
                continue
            m[r0:r1, c0:c1] = False
    m = ndimage.binary_erosion(m, iterations=4)
    sat = (rgb.max(axis=2) > 250) | (gray < 3)
    m &= ~sat
    return m.astype(np.float32)


def _wiener_wavelet_band(c, sigma2):
    best = None
    for w in (3, 5, 7, 9):
        loc = ndimage.uniform_filter(c * c, size=w, mode="reflect") - sigma2
        loc = np.maximum(loc, 0)
        best = loc if best is None else np.minimum(best, loc)
    return c * best / (best + sigma2)


def noise_extract_channel(x, sigma=SIGMA, levels=4, wavelet="db8"):
    h, w = x.shape
    coeffs = pywt.wavedec2(x, wavelet, level=levels, mode="periodization")
    s2 = sigma * sigma
    new = [np.zeros_like(coeffs[0])]  # residual: drop approximation
    for (cH, cV, cD) in coeffs[1:]:
        new.append(tuple(_wiener_wavelet_band(c, s2) for c in (cH, cV, cD)))
    # residual = detail coefficients removed by denoiser = c - wiener(c)
    res_coeffs = [np.zeros_like(coeffs[0])]
    for (orig, den) in zip(coeffs[1:], new[1:]):
        res_coeffs.append(tuple(o - d for o, d in zip(orig, den)))
    r = pywt.waverec2(res_coeffs, wavelet, mode="periodization")
    return r[:h, :w].astype(np.float32)


def zero_mean_total(r):
    r = r - r.mean(axis=0, keepdims=True)
    r = r - r.mean(axis=1, keepdims=True)
    return r


def wiener_dft(r, mask=None):
    """Whiten residual in Fourier domain (removes periodic JPEG/CFA artifacts)."""
    F = np.fft.fft2(r)
    mag = np.abs(F) / np.sqrt(r.size)
    sig2 = (mag ** 2).mean()  # noise variance estimate
    # local wiener on magnitude spectrum
    loc = ndimage.uniform_filter(mag ** 2, size=3, mode="wrap")
    att = np.maximum(loc - sig2, 0)
    Mag = mag * sig2 / (att + sig2)
    G = F * (Mag / (mag + 1e-12))
    return np.real(np.fft.ifft2(G)).astype(np.float32)


def residual_rgb(rgb, mask):
    """Per-channel residuals (zero-mean), plus gray-combined residual."""
    chans = []
    for c in range(3):
        r = noise_extract_channel(rgb[..., c])
        r = zero_mean_total(r * mask)
        chans.append(r)
    chans = np.stack(chans, 0)
    gray = 0.299 * chans[0] + 0.587 * chans[1] + 0.114 * chans[2]
    gray = zero_mean_total(gray) * mask
    return chans, gray.astype(np.float32)


def pce(a, b, excl=5):
    """Peak-to-correlation-energy of circular cross-correlation (peak searched at (0,0) +- 0)."""
    a = a - a.mean(); b = b - b.mean()
    xc = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    peak = xc[0, 0]
    h, w = xc.shape
    m = np.ones_like(xc, bool)
    for dy in range(-excl, excl + 1):
        for dx in range(-excl, excl + 1):
            m[dy % h, dx % w] = False
    energy = (xc[m] ** 2).mean()
    return float(np.sign(peak) * peak ** 2 / energy)
