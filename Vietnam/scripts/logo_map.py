"""Find where the brand evidence is: slide a 64x64 grey patch (stride 32) over each train image and record how much the
teacher's known-vs-negative margin drops. Saves a 7x7 drop map per image (logo_map.npy)."""
import csv, numpy as np, torch, timm, torch.nn.functional as F
from PIL import Image
dev = "cuda"
d = np.load("feats_e0_256.npz"); TB = torch.tensor(d["text_brand"]).float().to(dev); TE = torch.tensor(d["text_extra"]).float().to(dev)
TK, TN = TB[:3], torch.cat([TB[3:], TE])
t = timm.create_model("fastvit_mci0.apple_mclip2_dfndr2b", pretrained=True).to(dev).eval().half()
tr = list(csv.DictReader(open("data/train-output.csv"))); y = [["adidas", "converse", "nike"].index(r["label"]) for r in tr]
@torch.no_grad()
def margin(x, c):
    f = F.normalize(t(x.half()).float(), dim=-1)
    return (f @ TK[c]) - (f @ TN.t()).max(1).values
maps = np.zeros((len(tr), 7, 7), np.float32)
for i, r in enumerate(tr):
    x = torch.from_numpy(np.asarray(Image.open("data/train-input/" + r["filename"]).convert("RGB").resize((256, 256))).copy()).permute(2, 0, 1).float().div(255).to(dev)
    batch = [x]
    for gy in range(7):
        for gx in range(7):
            z = x.clone(); z[:, gy*32:gy*32+64, gx*32:gx*32+64] = 0.5; batch.append(z)
    m = margin(torch.stack(batch), y[i]).cpu().numpy()
    maps[i] = (m[0] - m[1:]).reshape(7, 7)
np.save("logo_map.npy", maps); print("saved", maps.shape, "mean max drop", maps.max((1, 2)).mean())
