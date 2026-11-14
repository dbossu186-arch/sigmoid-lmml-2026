"""GPU harness: try residual / weighting variants, LOO attribution accuracy on credited set."""
import sys, glob, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prnu_lib as P

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
data = sys.argv[1]
paths = sorted(glob.glob(f"{data}/credited/*/*.jpg"))
owners = np.array([p.split("/")[-2] for p in paths])
people = sorted(set(owners))
oid = np.array([people.index(o) for o in owners])

imgs, masks = [], []
for p in paths:
    rgb = P.load_rgb(p)
    imgs.append(torch.from_numpy(rgb).permute(2, 0, 1).half())
    masks.append(torch.from_numpy(P.caption_mask(rgb)).half())
imgs = torch.stack(imgs); masks = torch.stack(masks)
print("loaded", imgs.shape, flush=True)


def gk(s):
    r = max(1, int(3 * s + 0.5)); x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * s * s)); return k / k.sum()


def gblur(x, s):  # x: B,C,H,W
    k = gk(s); C = x.shape[1]
    x = F.conv2d(F.pad(x, (len(k) // 2,) * 2 + (0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0) + (len(k) // 2,) * 2, mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x


def med3(x):
    B, C, H, W = x.shape
    u = F.unfold(F.pad(x, (1, 1, 1, 1), mode="reflect").reshape(B * C, 1, H + 2, W + 2), 3)
    return u.median(1).values.view(B, C, H, W)


def zm(r):
    r = r - r.mean(-1, keepdim=True); return r - r.mean(-2, keepdim=True)


def residuals(kind, chan):
    out = []
    for i in range(len(paths)):
        x = imgs[i:i + 1].to(dev).float()
        if kind.startswith("hp"):
            r = x - gblur(x, float(kind[2:]))
        elif kind == "med3":
            r = x - med3(x)
        elif kind == "lap":
            r = x - 0.25 * (torch.roll(x, 1, -1) + torch.roll(x, -1, -1) + torch.roll(x, 1, -2) + torch.roll(x, -1, -2))
        if chan == "Y":
            r = (0.299 * r[:, 0] + 0.587 * r[:, 1] + 0.114 * r[:, 2])[:, None]; I = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
        elif chan == "RGB":
            I = x
        else:
            c = "RGB".index(chan); r = r[:, c:c + 1]; I = x[:, c:c + 1]
        m = masks[i][None, None].to(dev).float()
        out.append(((zm(r * m) * m).half().cpu(), (I * m).half().cpu()))
    return out


def evaluate(kind, chan, weight):
    R = residuals(kind, chan)
    num = torch.zeros((len(people),) + R[0][0].shape[1:], device=dev); den = torch.zeros_like(num)
    per = []
    for i, (r, I) in enumerate(R):
        r = r.to(dev).float(); I = I.to(dev).float()
        if weight == "texture":
            # down-weight textured areas: local energy of residual
            v = gblur(r * r, 4) + 1.0
            w = 1.0 / v
        else:
            w = torch.ones_like(r)
        a = (r * I * w)[0]; b = (I * I * w)[0]
        per.append(None)
        num[oid[i]] += a; den[oid[i]] += b; per[-1] = (a.half().cpu(), b.half().cpu())
    S = np.zeros((len(paths), len(people)))
    for i, (r, I) in enumerate(R):
        r = r.to(dev).float(); I = I.to(dev).float()
        for j in range(len(people)):
            n, d = num[j], den[j]
            if oid[i] == j:
                n = n - per[i][0].to(dev).float(); d = d - per[i][1].to(dev).float()
            K = zm(n / (d + 1.0))
            a = r[0]; b = (I[0] * K)
            if weight == "texture":
                ww = 1.0 / torch.sqrt(gblur(r * r, 4)[0] + 1.0); a = a * ww; b = b * ww
            a = a - a.mean(); b = b - b.mean()
            S[i, j] = float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
    pred = S.argmax(1); acc = (pred == oid).mean()
    # z-normalise score columns
    Sz = (S - S.mean(0)) / S.std(0); accz = (Sz.argmax(1) == oid).mean()
    ty, yk = people.index("toorisugari"), people.index("yakin")
    conf = sum((oid == ty) & (pred == yk)) + sum((oid == yk) & (pred == ty))
    print(f"{kind:6s} {chan:3s} {weight:8s} acc={acc:.3f} accz={accz:.3f} ty<->yk conf={conf}", flush=True)
    return S


if __name__ == "__main__":
    for kind in ["hp0.5", "hp0.7", "hp1", "med3", "lap"]:
        for chan in ["Y", "G", "RGB"]:
            for weight in ["none", "texture"]:
                evaluate(kind, chan, weight)
