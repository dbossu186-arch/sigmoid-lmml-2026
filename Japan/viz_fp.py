"""Build per-group fingerprints (texture-weighted MLE on hp0.5 residual) and visualise structure.
usage: viz_fp.py data outdir  [groups from credited folders + optional anon list file]"""
import sys, glob, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def _pick():
    import os
    if os.environ.get("DEV"): return os.environ["DEV"]
    if torch.backends.mps.is_available():
        return "mps"
    try:
        return "cuda" if torch.cuda.mem_get_info()[0] > 1.5e9 else "cpu"
    except Exception:
        return "cpu"
dev = _pick(); torch.set_num_threads(int(os.environ.get("NTHREADS", "2"))); print("device", dev, flush=True)


def gk(s):
    r = max(1, int(3 * s + 0.5)); x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * s * s)); return k / k.sum()


def gblur(x, s):
    k = gk(s); C = x.shape[1]; p = len(k) // 2
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x


def zm(r):
    r = r - r.mean(-1, keepdim=True); return r - r.mean(-2, keepdim=True)


def contrib(path, chan="RGB"):
    rgb = P.load_rgb(path); m = torch.from_numpy(P.caption_mask(rgb)).to(dev)[None, None]
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    r = x - gblur(x, 0.5)
    r = zm(r * m) * m
    w = 1.0 / (gblur(r * r, 4) + 1.0)
    I = x * m
    return (r * I * w)[0], (I * I * w)[0]


def fingerprint(paths):
    num = den = None
    for p in paths:
        a, b = contrib(p)
        num = a if num is None else num + a; den = b if den is None else den + b
    return zm(num / (den + 1.0)), den


if __name__ == "__main__":
    data, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    groups = {p: sorted(glob.glob(f"{data}/credited/{p}/*.jpg")) for p in
              ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]}
    if len(sys.argv) > 3:
        groups.update({k: [f"{data}/anonymous/{n}.jpg" for n in v] for k, v in json.load(open(sys.argv[3])).items()})
    Ks = {}
    for g, ps in groups.items():
        K, den = fingerprint(ps)
        Ks[g] = K
        Kg = K.mean(0, keepdim=True)[None]
        E = gblur(Kg * Kg, 6)[0, 0].cpu().numpy()          # local energy
        S = gblur(Kg, 3)[0, 0].cpu().numpy()                # low-pass of K (sign structure)
        np.save(f"{out}/K_{g}.npy", K.cpu().numpy().astype(np.float32))
        fig, ax = plt.subplots(1, 2, figsize=(20, 7.5))
        ax[0].imshow(E, cmap="gray", vmin=np.percentile(E, 1), vmax=np.percentile(E, 99)); ax[0].set_title(f"{g} local energy n={len(ps)}")
        ax[1].imshow(S, cmap="gray", vmin=np.percentile(S, 1), vmax=np.percentile(S, 99)); ax[1].set_title(f"{g} lowpass K")
        plt.tight_layout(); plt.savefig(f"{out}/viz_{g}.png", dpi=55); plt.close()
        # crop full-res
        c = K.mean(0)[300:428, 500:628].cpu().numpy()
        plt.imsave(f"{out}/crop_{g}.png", np.kron(c, np.ones((4, 4))), cmap="gray")
    names = list(Ks)
    for i, a in enumerate(names):
        print(a, " ".join(f"{b[:6]}={float(F.cosine_similarity(Ks[a].flatten(), Ks[b].flatten(), 0)):+.4f}" for b in names))
