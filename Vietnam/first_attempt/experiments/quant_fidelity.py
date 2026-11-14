"""Compare the exported int8 TorchScript model against the fp32 timm checkpoint on real images (CPU, Mac).
Usage: python quant_fidelity.py runs/final/model.pth model.pt [n_images]"""
import sys, csv, numpy as np, torch, timm, torch.nn.functional as F
from PIL import Image
torch.set_num_threads(4)
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
pt = torch.jit.load(sys.argv[2]).eval()
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
net = timm.create_model(ck["model"], pretrained=False, num_classes=ck["n_out"]).eval(); net.load_state_dict(ck["state_dict"])
rows = list(csv.DictReader(open("data/train-output.csv")))[:n]
X = torch.stack([torch.from_numpy(np.asarray(Image.open("data/train-input/" + r["filename"]).convert("RGB")).copy()).permute(2, 0, 1) for r in rows])
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
with torch.no_grad():
    x = (X.float() / 255 - mean) / std
    ref = (net(x) + net(torch.flip(x, [3]))) / 2
    rs = 1 - torch.softmax(ref, 1)[:, 3]
    out = torch.cat([pt(X[i:i + 50]) for i in range(0, len(X), 50)])
thr = float(pt.thr)
ref_pred = torch.where(rs < thr, torch.tensor(3), ref[:, :3].argmax(1))
print(f"n={len(X)} thr={thr}: decision agreement fp32 vs int8 = {(ref_pred == out.argmax(1)).float().mean():.4f}; "
      f"known-logprob max|diff| = {(torch.log_softmax(ref, 1)[:, :3] - out[:, :3]).abs().max():.4f}")
