"""Export a trained run (model.pth from train.py) to a self-contained TorchScript model.pt.

* Weights: per-output-channel symmetric int8 for every >=2-D float tensor (conv/linear),
  fp16 for 1-D tensors (BN / bias); dequantised on the fly inside the traced graph.
* Interface (scripted wrapper):
    input  : [B,3,H,W] or [3,H,W], uint8 or float (divided by 255 if max > 1.5),
             any H,W (resized internally to the training size), channels-last [B,H,W,3] also accepted
    output : [B,4] logits in order classes = ['adidas','converse','nike','unknown']
             first 3 = log-probabilities of the known brands,
             4th     = max(known) + K*(thr - s) with s = 1 - p(unknown)   (the open-set score)
             => argmax == 'unknown'  iff  s < thr.
Run with the pinned platform versions (torch 2.11 / torchvision 0.26) and timm==1.0.31 (export-time only).
Usage: python export.py runs/final/model.pth --thr 0.5 --out model.pt
"""
import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, timm
from typing import List

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--thr", type=float, required=True, help="reject when 1 - p(unknown) < thr")
p.add_argument("--out", default="model.pt")
p.add_argument("--wq", default="fp16", choices=["fp16", "int8", "gint8"],
               help="storage of >=2-D weights: fp16, per-channel int8, or group-wise int8 (groups of 32)")
args = p.parse_args()
torch.manual_seed(0)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
assert ck["known"] == ["adidas", "converse", "nike"] and ck["n_out"] == 4
net = timm.create_model(ck["model"], pretrained=False, num_classes=ck["n_out"]).eval()
net.load_state_dict(ck["state_dict"])
SIZE = int(ck["size"])

class QuantInner(nn.Module):
    """Holds compressed weights; on forward, dequantises them and swaps them into the network."""
    def __init__(self, m: nn.Module, wq: str, G: int = 32):
        super().__init__()
        sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        self.m, self.names, self.shapes = m, list(sd.keys()), {}
        for i, (k, v) in enumerate(sd.items()):
            if v.dtype.is_floating_point and v.dim() >= 2 and wq == "int8":
                shape = (-1,) + (1,) * (v.dim() - 1)
                s = (v.abs().flatten(1).amax(1).clamp_min(1e-12) / 127.0).view(shape)
                self.register_buffer(f"q{i}", torch.round(v / s).clamp(-127, 127).to(torch.int8))
                self.register_buffer(f"s{i}", s.float())
            elif v.dtype.is_floating_point and v.dim() >= 2 and wq == "gint8":
                w2 = v.flatten(1); n_in = w2.shape[1]; ng = (n_in + G - 1) // G
                w3 = F.pad(w2, (0, ng * G - n_in)).view(w2.shape[0], ng, G)
                s = (w3.abs().amax(2, keepdim=True).clamp_min(1e-12) / 127.0).half().float()
                self.register_buffer(f"g{i}", torch.round(w3 / s).clamp(-127, 127).to(torch.int8))
                self.register_buffer(f"s{i}", s.half())
                self.shapes[i] = (list(v.shape), n_in)
            elif v.dtype.is_floating_point:
                self.register_buffer(f"f{i}", v.half() if (wq == "fp16" and v.dim() >= 2) else v.float())
            else:
                self.register_buffer(f"i{i}", v)
        for mod in m.modules():  # drop the fp32 originals (they are re-injected at forward time)
            for n, t in list(mod._parameters.items()):
                if t is not None: mod._parameters[n] = nn.Parameter(torch.empty(0), requires_grad=False)
            for n, t in list(mod._buffers.items()):
                if t is not None: mod._buffers[n] = torch.empty(0, dtype=t.dtype)

    def forward(self, x):
        for i, k in enumerate(self.names):
            if hasattr(self, f"q{i}"): v = getattr(self, f"q{i}").float() * getattr(self, f"s{i}")
            elif hasattr(self, f"g{i}"):
                shape, n_in = self.shapes[i]
                g = getattr(self, f"g{i}").float() * getattr(self, f"s{i}").float()
                v = g.reshape(shape[0], -1)[:, :n_in].reshape(shape)
            elif hasattr(self, f"f{i}"): v = getattr(self, f"f{i}").float()
            else: v = getattr(self, f"i{i}")
            *path, attr = k.split(".")
            mod = self.m.get_submodule(".".join(path))
            if attr in mod._parameters: mod._parameters[attr] = v
            else: mod._buffers[attr] = v
        return self.m(x)

class SneakerOSR(nn.Module):
    classes: List[str]
    def __init__(self, inner: torch.jit.ScriptModule, thr: float, size: int):
        super().__init__()
        self.inner = inner
        self.classes = ["adidas", "converse", "nike", "unknown"]
        self.thr, self.size, self.k = float(thr), int(size), 20.0
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != 3 and x.shape[-1] == 3:  # channels-last input
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = x[:, :3].float()
        if bool(x.max() > 1.5):
            x = x / 255.0
        if x.shape[-2] != self.size or x.shape[-1] != self.size:
            x = F.interpolate(x, size=[self.size, self.size], mode="bicubic", align_corners=False, antialias=True)
            x = x.clamp(0.0, 1.0)
        x = (x - self.mean) / self.std
        logits = (self.inner(x) + self.inner(torch.flip(x, [3]))) / 2.0
        logp = torch.log_softmax(logits, dim=1)
        known = logp[:, :3]
        s = 1.0 - torch.exp(logp[:, 3])
        mk = known.max(dim=1).values
        unk = mk + self.k * (self.thr - s)
        return torch.cat([known, unk.unsqueeze(1)], dim=1)

if __name__ == "__main__":
    ref_in = torch.rand(2, 3, SIZE, SIZE)
    inner = QuantInner(net, args.wq).eval()
    with torch.no_grad():
        traced = torch.jit.trace(inner, ref_in, check_trace=False)
    model = torch.jit.script(SneakerOSR(traced, args.thr, SIZE).eval())
    torch.jit.save(model, args.out)
    print(f"saved {args.out}: {os.path.getsize(args.out)/1e6:.2f} MB, thr={args.thr}, base={ck['model']}, size={SIZE}")
