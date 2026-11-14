"""Visualise CAM-erased 'evidence-removed' outliers for a trained checkpoint (Mac, CPU)."""
import sys, csv, numpy as np, torch, timm, torch.nn.functional as F
from PIL import Image
torch.set_num_threads(4)
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = timm.create_model(ck["model"], num_classes=ck["n_out"]).eval(); m.load_state_dict(ck["state_dict"])
area = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
rows = list(csv.DictReader(open("data/train-output.csv")))[100:112]
CLS = ["adidas", "converse", "nike"]
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
X = torch.stack([torch.from_numpy(np.asarray(Image.open("data/train-input/" + r["filename"]).convert("RGB")).copy()).permute(2, 0, 1) for r in rows]).float() / 255
x = (X - mean) / std; y = torch.tensor([CLS.index(r["label"]) for r in rows])
with torch.no_grad():
    fm = m.forward_features(x); W = m.get_classifier().weight
    cam = F.interpolate(torch.einsum("bchw,bc->bhw", fm, W[y]).unsqueeze(1), size=x.shape[-2:], mode="bilinear")
    q = torch.quantile(cam.flatten(1), 1 - area, dim=1).view(-1, 1, 1, 1); mk = (cam >= q).float()
    mk = F.avg_pool2d(F.max_pool2d(mk, 9, 1, 4), 9, 1, 4)
    blur = F.avg_pool2d(F.avg_pool2d(x, 15, 1, 7, count_include_pad=False), 15, 1, 7, count_include_pad=False)
    xe = x * (1 - mk) + blur * mk
    p0 = torch.softmax(m(x), 1); p1 = torch.softmax(m(xe), 1)
img = lambda t: Image.fromarray((((t * std[0] + mean[0]).clamp(0, 1)).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((160, 160))
G = Image.new("RGB", (160 * 6, 160 * 4))
for i in range(12):
    G.paste(img(x[i]), ((i % 6) * 160, (i // 6) * 320)); G.paste(img(xe[i]), ((i % 6) * 160, (i // 6) * 320 + 160))
    print(rows[i]["label"], "p_unk before %.2f after %.2f" % (p0[i, 3], p1[i, 3]))
G.save(sys.argv[2]); print("saved")
