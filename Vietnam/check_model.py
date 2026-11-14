"""Load-and-predict test for model.pt with BOTH torch.jit.load and torch.load(weights_only=False).
Run under the pinned platform env: torch==2.11.0 torchvision==0.26.0 numpy==2.4.6 Pillow==12.3.0 (CPU).
Usage: python check_model.py [model.pt] [data_dir]
"""
import sys, os, csv, warnings, numpy as np, torch
from PIL import Image
path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
data = sys.argv[2] if len(sys.argv) > 2 else "data"
torch.set_num_threads(4)
print("torch", torch.__version__, "| file size %.2f MB" % (os.path.getsize(path) / 1e6))
assert os.path.getsize(path) <= 10 * 1024 * 1024, "model.pt exceeds 10 MB"

m1 = torch.jit.load(path, map_location="cpu").eval()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m2 = torch.load(path, weights_only=False, map_location="cpu").eval()
classes = list(m1.classes)
print("classes:", classes)
assert classes == ["adidas", "converse", "nike", "unknown"] and list(m2.classes) == classes

rows = list(csv.DictReader(open(f"{data}/train-output.csv")))[:16]
imgs = [torch.from_numpy(np.asarray(Image.open(f"{data}/train-input/{r['filename']}").convert("RGB")).copy()).permute(2, 0, 1) for r in rows]
xb = torch.stack(imgs)  # uint8 [B,3,256,256]
with torch.no_grad():
    o1 = m1(xb); o2 = m2(xb)
    o_float = m1(xb.float() / 255)            # float in [0,1]
    o_single = torch.stack([m1(im)[0] for im in imgs[:4]])  # [3,H,W] uint8
    o_resized = m1(torch.nn.functional.interpolate(xb[:4].float(), size=(300, 200), mode="bilinear"))  # other size, float 0-255
assert o1.shape == (len(rows), 4), o1.shape
print("jit.load vs torch.load max|diff| =", (o1 - o2).abs().max().item())
print("uint8 vs float[0,1] max|diff|   =", (o1 - o_float).abs().max().item())
print("batch vs single max|diff|       =", (o1[:4] - o_single).abs().max().item())
assert torch.equal(o1, o2)
assert torch.equal(o1.argmax(1), o_float.argmax(1)) and torch.equal(o1[:4].argmax(1), o_single.argmax(1))
pred = [classes[i] for i in o1.argmax(1).tolist()]
gt = [r["label"] for r in rows]
print("predictions on 16 TRAIN images (sanity only, these were seen in training):")
print("  acc=%.3f  rejected=%d  other-size preds=%s" % (np.mean([p == g for p, g in zip(pred, gt)]), pred.count("unknown"),
      [classes[i] for i in o_resized.argmax(1).tolist()]))
ref = sys.argv[3] if len(sys.argv) > 3 else "head_ref_decisions.npy"
if os.path.exists(ref):  # decisions of the head computed at train time (GPU) on all 735 training photos
    rows = list(csv.DictReader(open(f"{data}/train-output.csv"))); dec = []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            xb = torch.stack([torch.from_numpy(np.asarray(Image.open(f"{data}/train-input/{r['filename']}").convert("RGB")).copy()).permute(2, 0, 1) for r in rows[i:i + 16]])
            dec.append(m1(xb).argmax(1))
    dec = torch.cat(dec).numpy(); refd = np.load(ref)
    gt = np.array([classes.index(r["label"]) for r in rows])
    print(f"all {len(rows)} train photos: agreement with train-time decisions = {np.mean(dec == refd):.4f}; "
          f"rejected = {np.mean(dec == 3):.3f}; accepted-and-correct = {np.mean(dec == gt):.3f}")
print("OK")
