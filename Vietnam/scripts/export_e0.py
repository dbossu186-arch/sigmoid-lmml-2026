"""Zero-shot open-set sneaker model: MobileCLIP2-S0 image tower (timm, reparameterized, int8/int4 weights) +
frozen text prototypes (3 known brands vs ~75 negative brand / 'fake' prompts). No training on the images.
Decision: margin = max_k cos(img, known_k) - max_j cos(img, neg_j);  'unknown' iff margin < thr.
Output [B,4] logits, classes = ['adidas','converse','nike','unknown'].
Usage: python export_e0.py feats.npz --thr -0.026 --out model.pt [--int4 200000]"""
import argparse, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from typing import List
from timm.utils.model import reparameterize_model
sys.path.insert(0, os.path.dirname(__file__))
from osr_export_lib_copy import QuantInner, size_policy
p = argparse.ArgumentParser(); p.add_argument("npz"); p.add_argument("--thr", type=float, required=True)
p.add_argument("--out", default="model.pt"); p.add_argument("--int4", type=int, default=200000); p.add_argument("--noreparam", action="store_true")
p.add_argument("--G", type=int, default=32)
p.add_argument("--negset", default="all", choices=["all", "brands"])
a = p.parse_args(); torch.manual_seed(0)
d = np.load(a.npz)
TB, TE = torch.tensor(d["text_brand"]).float(), torch.tensor(d["text_extra"]).float()
NEG = torch.cat([TB[3:], TE]) if a.negset == "all" else TB[3:]
net = timm.create_model("fastvit_mci0.apple_mclip2_dfndr2b", pretrained=True).eval()
if not a.noreparam: net = reparameterize_model(net)
SIZE = 256

class ZSOSR(nn.Module):
    classes: List[str]
    def __init__(self, inner, known, neg, thr: float, scale: float):
        super().__init__()
        self.inner = inner
        self.classes = ["adidas", "converse", "nike", "unknown"]
        self.register_buffer("known", known.clone()); self.register_buffer("neg", neg.clone())
        self.thr, self.scale, self.size = float(thr), float(scale), SIZE
        self.register_buffer("in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        f = F.normalize(self.inner(x), dim=-1) + F.normalize(self.inner(torch.flip(x, [3])), dim=-1)
        return F.normalize(f, dim=-1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3: x = x.unsqueeze(0)
        if x.shape[1] != 3 and x.shape[-1] == 3: x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1: x = x.expand(-1, 3, -1, -1)
        x = x[:, :3].float()
        if bool(x.max() > 1.5): x = x / 255.0
        elif bool(x.min() < -0.05): x = x * self.in_std + self.in_mean   # ImageNet-normalised input -> [0,1]
        if x.shape[-2] != self.size or x.shape[-1] != self.size:
            x = F.interpolate(x, size=[self.size, self.size], mode="bicubic", align_corners=False, antialias=True)
        x = x.clamp(0.0, 1.0)
        f = self.embed(x)
        k = f @ self.known.t(); n = (f @ self.neg.t()).max(dim=1).values
        unk = n + self.thr
        return self.scale * torch.cat([k, unk.unsqueeze(1)], dim=1)

def polA(k, v):  # int4 (G16) for the 7 largest matrices + stage-2 MLPs; int8 per-channel elsewhere (downsample convs are int4-sensitive)
    return "int4" if (v.numel() >= 400000 or ("stages.2" in k and "mlp.fc" in k)) else "int8"
inner = QuantInner(net, polA if a.int4 == -1 else size_policy(a.int4), G=a.G).eval() if a.int4 != 0 else net
with torch.no_grad():
    traced = torch.jit.trace(inner, torch.rand(2, 3, SIZE, SIZE), check_trace=False)
model = torch.jit.script(ZSOSR(traced, TB[:3], NEG, a.thr, float(d["logit_scale"])).eval())
torch.jit.save(model, a.out)
print(f"saved {a.out}: {os.path.getsize(a.out)/1e6:.2f} MB ({os.path.getsize(a.out)/2**20:.2f} MiB) thr={a.thr} int4>={a.int4} reparam={not a.noreparam}")
