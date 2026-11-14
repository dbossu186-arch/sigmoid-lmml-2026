"""Export the submission model.pt (TorchScript, no code needed at load time).

Stage 1  python export.py backbone --out backbone.pt
         MobileCLIP2-S0 image tower (timm `fastvit_mci0.apple_mclip2_dfndr2b`, reparameterised), weights compressed
         (group-wise int4 for transformer/ConvMLP linears, per-channel int8 elsewhere, fp32 norms), traced.
Stage 2  python export.py final --backbone backbone.pt --head head.npz --out model.pt
         Scripted wrapper = preprocessing + backbone (flip-TTA) + open-set head from train.py.

Interface of model.pt
  input : [B,3,H,W] or [3,H,W]; uint8 or float (divided by 255 if max > 1.5); any H,W (bicubic-resized to 256);
          channels-last [B,H,W,3] and 1-channel inputs are also accepted.
  output: [B,4] logits, order model.classes = ['adidas','converse','nike','unknown'].
          logits[:, :3] = log-softmax of the brand classifier;
          logits[:, 3]  = max(logits[:, :3]) + 20 * (thr - s), s = open-set score (higher = more 'known')
          => argmax == 'unknown' exactly when s < thr. A plain argmax gives the final label.
Run under the pinned platform versions (torch 2.11.0 / torchvision 0.26.0); timm==1.0.31 is needed only for stage 1.
"""
import argparse, os
from typing import List
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

BACKBONE = "fastvit_mci0.apple_mclip2_dfndr2b"


class SneakerOSR(nn.Module):
    classes: List[str]

    def __init__(self, backbone: torch.jit.ScriptModule, head: dict):
        super().__init__()
        self.backbone = backbone
        self.classes = ["adidas", "converse", "nike", "unknown"]
        self.size = 256
        self.k = int(head["k"])
        st = head["stats"].astype(np.float64)
        self.mk, self.sk, self.ml, self.sl = float(st[0]), float(st[1]), float(st[2]), float(st[3])
        self.thr = float(head["thr"])
        self.margin = 20.0
        self.register_buffer("W", torch.from_numpy(head["W"]).float())            # [3, D]
        self.register_buffer("b", torch.from_numpy(head["b"]).float())            # [3]
        self.register_buffer("bank_q", torch.from_numpy(head["bank_q"]))          # [N, D] int8
        self.register_buffer("bank_s", torch.from_numpy(head["bank_s"]).float())  # [N]
        self.register_buffer("bank_y", torch.from_numpy(head["bank_y"]).long())   # [N]

    def scores(self, x: torch.Tensor):
        """Returns (brand logits [B,3], open-set score s [B])."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != 3 and x.shape[-1] == 3:      # channels-last
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = x[:, :3].float()
        if bool(x.max() > 1.5):
            x = x / 255.0
        if x.shape[-2] != self.size or x.shape[-1] != self.size:
            x = F.interpolate(x, size=[self.size, self.size], mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
        f = (self.backbone(x) + self.backbone(torch.flip(x, [3]))) / 2.0
        z = F.normalize(f, dim=1)
        logits = z @ self.W.t() + self.b
        pred = logits.argmax(dim=1)
        bank = self.bank_q.float() * self.bank_s.unsqueeze(1)
        sims = z @ bank.t()
        same = self.bank_y.unsqueeze(0) == pred.unsqueeze(1)
        sims = torch.where(same, sims, torch.full_like(sims, -2.0))
        knn = sims.topk(self.k, dim=1).values.mean(dim=1)
        s = (knn - self.mk) / self.sk + (logits.max(dim=1).values - self.ml) / self.sl
        return logits, s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, s = self.scores(x)
        known = torch.log_softmax(logits, dim=1)
        unk = known.max(dim=1).values + self.margin * (self.thr - s)
        return torch.cat([known, unk.unsqueeze(1)], dim=1)


def build_backbone(out: str):
    import timm
    from timm.utils.model import reparameterize_model
    from osr_export_lib import QuantInner, mci_policy
    torch.manual_seed(0)
    m = reparameterize_model(timm.create_model(BACKBONE, pretrained=True, num_classes=0).eval())
    q = QuantInner(m, mci_policy("int8")).eval()
    with torch.no_grad():
        tr = torch.jit.trace(q, torch.rand(2, 3, 256, 256), check_trace=False)
    torch.jit.save(tr, out)
    print(f"saved {out}: {os.path.getsize(out)/1e6:.2f} MB")


def build_final(backbone: str, head: str, out: str):
    bb = torch.jit.load(backbone, map_location="cpu").eval()
    h = dict(np.load(head))
    model = torch.jit.script(SneakerOSR(bb, h).eval())
    torch.jit.save(model, out)
    print(f"saved {out}: {os.path.getsize(out)/1e6:.3f} MB ({os.path.getsize(out)} bytes), thr={float(h['thr']):.4f}, bank={h['bank_q'].shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["backbone", "final"])
    ap.add_argument("--backbone", default="backbone.pt")
    ap.add_argument("--head", default="head.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.stage == "backbone":
        build_backbone(a.out or "backbone.pt")
    else:
        build_final(a.backbone, a.head, a.out or "model.pt")
