"""Weight compression + tracing helpers shared by export.py and the fidelity checks.

QuantInner stores every >=2-D float weight compressed and re-injects the dequantised tensors into the
network at forward time, so the traced TorchScript graph contains only the compressed buffers:
  * 'int8' : per-output-channel symmetric int8
  * 'int4' : group-wise (groups of G along the flattened input dim) symmetric int4, two values packed per byte
  * 'fp16' : half precision
1-D tensors (norm / bias) stay fp32.
"""
from typing import Callable, Dict, List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F


class QuantInner(nn.Module):
    def __init__(self, m: nn.Module, policy: Callable[[str, torch.Tensor], str], G: int = 32, forward_fn: str = "forward"):
        super().__init__()
        sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        self.m, self.names, self.meta, self.forward_fn = m, list(sd.keys()), {}, forward_fn
        self.kinds: Dict[str, int] = {}
        for i, (k, v) in enumerate(sd.items()):
            kind = policy(k, v) if (v.dtype.is_floating_point and v.dim() >= 2) else ("f32" if v.dtype.is_floating_point else "raw")
            self.kinds[k] = 0
            if kind == "int8":
                shape = (-1,) + (1,) * (v.dim() - 1)
                s = (v.abs().flatten(1).amax(1).clamp_min(1e-12) / 127.0).view(shape)
                self.register_buffer(f"q{i}", torch.round(v / s).clamp(-127, 127).to(torch.int8))
                self.register_buffer(f"s{i}", s.float())
            elif kind == "int4":
                w2 = v.flatten(1); n_in = w2.shape[1]; ng = (n_in + G - 1) // G
                w3 = F.pad(w2, (0, ng * G - n_in)).view(w2.shape[0], ng, G)
                s = (w3.abs().amax(2, keepdim=True).clamp_min(1e-12) / 7.0).half().float()
                q = (torch.round(w3 / s).clamp(-7, 7) + 8).to(torch.uint8).view(w2.shape[0], ng, G // 2, 2)
                self.register_buffer(f"p{i}", (q[..., 0] << 4) | q[..., 1])          # packed nibbles
                self.register_buffer(f"s{i}", s.half())
                self.meta[i] = (list(v.shape), n_in, G)
            elif kind == "fp16":
                self.register_buffer(f"h{i}", v.half())
            elif kind == "f32":
                self.register_buffer(f"f{i}", v.float())
            else:
                self.register_buffer(f"r{i}", v)
        for mod in m.modules():  # drop the fp32 originals
            for n, t in list(mod._parameters.items()):
                if t is not None: mod._parameters[n] = nn.Parameter(torch.empty(0), requires_grad=False)
            for n, t in list(mod._buffers.items()):
                if t is not None: mod._buffers[n] = torch.empty(0, dtype=t.dtype)

    def dequant(self, i: int) -> torch.Tensor:
        if hasattr(self, f"q{i}"): return getattr(self, f"q{i}").float() * getattr(self, f"s{i}")
        if hasattr(self, f"p{i}"):
            shape, n_in, G = self.meta[i]
            pk = getattr(self, f"p{i}")
            q = torch.stack([(pk >> 4) & 15, pk & 15], dim=-1).reshape(pk.shape[0], pk.shape[1], G).float() - 8.0
            return (q * getattr(self, f"s{i}").float()).reshape(shape[0], -1)[:, :n_in].reshape(shape)
        if hasattr(self, f"h{i}"): return getattr(self, f"h{i}").float()
        if hasattr(self, f"f{i}"): return getattr(self, f"f{i}")
        return getattr(self, f"r{i}")

    def forward(self, x):
        for i, k in enumerate(self.names):
            v = self.dequant(i)
            *path, attr = k.split(".")
            mod = self.m.get_submodule(".".join(path))
            if attr in mod._parameters: mod._parameters[attr] = v
            else: mod._buffers[attr] = v
        return getattr(self.m, self.forward_fn)(x)


def size_policy(int4_min_numel: int = 200_000, int8: bool = True):
    """int4 for the largest matrices, int8 (or fp16) for the rest."""
    def pol(k: str, v: torch.Tensor) -> str:
        if v.numel() >= int4_min_numel: return "int4"
        return "int8" if int8 else "fp16"
    return pol


def mci_policy(rest: str = "fp16"):
    """MobileCLIP/FastViT: int4 (g=32) only for transformer/ConvMLP linear weights, which tolerate it;
    downsample / stem / reparam convs are very sensitive and stay fp16 (or int8)."""
    def pol(k: str, v: torch.Tensor) -> str:
        if v.numel() >= 100_000 and (".mlp.fc" in k or ".token_mixer.qkv" in k or ".token_mixer.proj" in k):
            return "int4"
        return rest
    return pol
