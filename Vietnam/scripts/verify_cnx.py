"""Grader-style check: default torch.load (weights_only=True), strict load into torchvision convnext_tiny(len(classes)),
preprocess = ToTensor + Normalize(mean,std) at native 256x256, rule: max softmax < tau -> 'unknown'.
Usage: python verify_cnx.py model.pt [holdout_idx.npy] [student.pth for fp32 fidelity]"""
import sys, os, csv, numpy as np, torch, torchvision
from PIL import Image
torch.set_num_threads(4)
path = sys.argv[1]; sz = os.path.getsize(path)
print(f"torch {torch.__version__} tv {torchvision.__version__} | {sz:,} bytes = {sz/1e6:.3f} MB = {sz/2**20:.3f} MiB | under 10e6: {sz < 10_000_000}")
ck = torch.load(path, map_location="cpu")
C, tau = ck["classes"], ck["tau"]; print("classes", C, "tau", tau, "mean", ck["mean"], "std", ck["std"])
m = torchvision.models.convnext_tiny(num_classes=len(C)); m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()}); m.eval()
print("strict load OK")
mean = torch.tensor(ck["mean"]).view(1, 3, 1, 1); std = torch.tensor(ck["std"]).view(1, 3, 1, 1)
def probs(paths):
    out = []
    for i in range(0, len(paths), 32):
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()).permute(2, 0, 1) for p in paths[i:i+32]]).float() / 255
        with torch.no_grad(): out.append(torch.softmax(m((x - mean) / std), 1))
    return torch.cat(out).numpy()
tr = list(csv.DictReader(open("data/train-output.csv")))
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    idx = np.load(sys.argv[2]); rows = [tr[i] for i in idx]; tag = "HELD-OUT real (never trained on)"
else: rows = tr; tag = "ALL real train (seen in training -> optimistic)"
pv = probs(["data/train-input/" + r["filename"] for r in rows]); yv = np.array([C.index(r["label"]) for r in rows])
z = list(csv.DictReader(open("zsub/zsub.csv")))
pz = probs(["zsub/" + r["path"] for r in z]); lab = [r["label"] for r in z]
ki = np.array([l != "unknown" for l in lab]); zky = np.array([C.index(l) if l != "unknown" else -1 for l in lab])
for t in sorted({tau} | {round(0.55 + 0.02 * i, 2) for i in range(12)}):
    s = "  <== shipped tau" if t == tau else ""
    print(f"tau={t:.2f}: {tag}: known acc={np.mean((pv.argmax(1)==yv)&(pv.max(1)>=t)):.3f} (n={len(yv)}) | Zappos known acc={np.mean((pz[ki].argmax(1)==zky[ki])&(pz[ki].max(1)>=t)):.3f} (n={ki.sum()}) rejection={np.mean(pz[~ki].max(1)<t):.3f} (n={(~ki).sum()}){s}")
