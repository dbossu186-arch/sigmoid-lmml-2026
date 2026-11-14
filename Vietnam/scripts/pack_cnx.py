"""Pack a pruned student into the grader's plain-dict format.
Dropped blocks: layer_scale = 0 and every tensor stored as a stride-0 expanded zero (~1 element on disk).
Kept blocks: both pointwise Linears stored as int8 with the per-channel scale folded exactly into neighbours:
  pw2 (block.5): W = s_out*Q  -> layer_scale *= s_out, bias /= s_out
  pw1 (block.3): W = Q*s_in   -> preceding LayerNorm weight,bias *= s_in
  downsample convs: same trick on the input channels, folded into the LayerNorm2d before them
Everything else >=2-D: fp16. 1-D / small: fp32.  Grader does {k: v.float()} then strict load into convnext_tiny(3).
Usage: python pack_cnx.py student.pth --tau 0.7 --out model.pt"""
import argparse, os, torch, torchvision
p = argparse.ArgumentParser(); p.add_argument("ckpt"); p.add_argument("--tau", type=float, required=True); p.add_argument("--out", required=True)
p.add_argument("--noint8", action="store_true"); a = p.parse_args()
st = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
tmpl = torchvision.models.convnext_tiny(num_classes=3).state_dict()
out = {}
def zero_like(v): return torch.zeros(1, dtype=torch.float16).expand(*v.shape)
blocks = sorted({k.rsplit(".block.", 1)[0] for k in tmpl if ".block." in k})
for k, v in tmpl.items():
    if k not in st: out[k] = zero_like(v); continue
    w = st[k].float()
    out[k] = w.half() if w.dim() >= 2 else w.clone()
if not a.noint8:
    for b in blocks:
        if b + ".block.5.weight" not in st: continue
        W2, b2, g = st[b + ".block.5.weight"].float(), st[b + ".block.5.bias"].float(), st[b + ".layer_scale"].float()
        s = W2.abs().amax(1).clamp_min(1e-12) / 127.0
        out[b + ".block.5.weight"] = torch.round(W2 / s[:, None]).clamp(-127, 127).to(torch.int8)
        out[b + ".block.5.bias"] = b2 / s; out[b + ".layer_scale"] = g * s.view(-1, 1, 1)
        W1, lg, lb = st[b + ".block.3.weight"].float(), st[b + ".block.2.weight"].float(), st[b + ".block.2.bias"].float()
        s1 = W1.abs().amax(0).clamp_min(1e-12) / 127.0
        out[b + ".block.3.weight"] = torch.round(W1 / s1[None, :]).clamp(-127, 127).to(torch.int8)
        out[b + ".block.2.weight"] = lg * s1; out[b + ".block.2.bias"] = lb * s1
    # downsample convs (features.{2,4,6}.1, 2x2 stride 2): per-input-channel scale folded into the LayerNorm2d before them
    for i in (2, 4, 6):
        Wd, lg, lb = st[f"features.{i}.1.weight"].float(), st[f"features.{i}.0.weight"].float(), st[f"features.{i}.0.bias"].float()
        sd_ = Wd.abs().amax(dim=(0, 2, 3)).clamp_min(1e-12) / 127.0
        out[f"features.{i}.1.weight"] = torch.round(Wd / sd_.view(1, -1, 1, 1)).clamp(-127, 127).to(torch.int8)
        out[f"features.{i}.0.weight"] = lg * sd_; out[f"features.{i}.0.bias"] = lb * sd_
ck = {"classes": ["adidas", "converse", "nike"], "tau": float(a.tau), "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "state_dict": out}
torch.save(ck, a.out)
print(f"saved {a.out}: {os.path.getsize(a.out):,} bytes ({os.path.getsize(a.out)/1e6:.3f} MB) tau={a.tau}")
