"""Standalone load + eval of a .pt in the pinned env (torch 2.11/tv 0.26). Only torch/numpy/PIL used.
Usage: python eval_pt.py model.pt"""
import sys, os, csv, warnings, numpy as np, torch
from PIL import Image
torch.set_num_threads(4)
path = sys.argv[1]
print("torch", torch.__version__, "| size %.3f MB / %.3f MiB" % (os.path.getsize(path) / 1e6, os.path.getsize(path) / 2**20))
m = torch.jit.load(path, map_location="cpu").eval()
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); m2 = torch.load(path, weights_only=False, map_location="cpu").eval()
C = list(m.classes); print("classes", C)
def run(paths):
    out = []
    for i in range(0, len(paths), 32):
        xb = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()).permute(2, 0, 1) for p in paths[i:i+32]])
        with torch.no_grad(): out.append(m(xb))
    return torch.cat(out)
tr = list(csv.DictReader(open("data/train-output.csv")))
o = run(["data/train-input/" + r["filename"] for r in tr]); pred = [C[i] for i in o.argmax(1).tolist()]
gt = [r["label"] for r in tr]
print("REAL train (zero-shot, never trained on): known acc=%.3f  rejected=%.3f" % (np.mean([p == g for p, g in zip(pred, gt)]), np.mean([p == "unknown" for p in pred])))
for b in ["adidas", "converse", "nike"]:
    idx = [i for i, g in enumerate(gt) if g == b]; print("   ", b, "acc=%.3f" % np.mean([pred[i] == b for i in idx]))
z = list(csv.DictReader(open("zsub/zsub.csv")))
oz = run(["zsub/" + r["path"] for r in z]); pz = [C[i] for i in oz.argmax(1).tolist()]
k = [(p, r["label"]) for p, r in zip(pz, z) if r["label"] != "unknown"]; u = [p for p, r in zip(pz, z) if r["label"] == "unknown"]
print("ZAPPOS sub: known acc=%.3f (n=%d)  rejection(Puma/Reebok/Vans/Asics)=%.3f (n=%d)" % (np.mean([p == g for p, g in k]), len(k), np.mean([p == "unknown" for p in u]), len(u)))
xb = torch.stack([torch.from_numpy(np.asarray(Image.open("data/train-input/" + r["filename"]).convert("RGB")).copy()).permute(2, 0, 1) for r in tr[:8]])
with torch.no_grad():
    a, b2 = m(xb), m2(xb); f = m(xb.float() / 255); s = m(xb[0]); rs = m(torch.nn.functional.interpolate(xb.float(), size=(300, 200), mode="bilinear"))
    nrm = m((xb.float() / 255 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
print("jit vs torch.load equal:", torch.equal(a, b2), "| uint8 vs float01 argmax same:", torch.equal(a.argmax(1), f.argmax(1)),
      "| single-image shape", tuple(s.shape), "| imagenet-normalised argmax same:", torch.equal(a.argmax(1), nrm.argmax(1)), "| 300x200 preds", [C[i] for i in rs.argmax(1).tolist()])
