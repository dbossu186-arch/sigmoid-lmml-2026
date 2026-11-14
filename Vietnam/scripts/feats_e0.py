"""Extract MobileCLIP2-S0 image embeddings (timm, 512-d CLIP space) + text embeddings (open_clip) for OSR analysis."""
import os, csv, sys, numpy as np, torch, timm, torch.nn.functional as F
from PIL import Image
torch.set_num_threads(2)
dev = "cuda"
CLS = ["adidas", "converse", "nike"]
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 256
tr = list(csv.DictReader(open("data/train-output.csv")))
zi = list(csv.DictReader(open("ext/zappos_index.csv")))
sets = {
 "real": ["data/train-input/" + r["filename"] for r in tr],
 "zk": [r["path"] for r in zi if r["role"] == "val_known" and r["split"] == "val"],
 "zu": [r["path"] for r in zi if r["role"] == "val_unk"],
 "zo": [r["path"] for r in zi if r["role"] == "oe"][::4],
}
meta = {"real_y": np.array([CLS.index(r["label"]) for r in tr]),
        "zk_y": np.array([CLS.index(r["label"]) for r in zi if r["role"] == "val_known" and r["split"] == "val"]),
        "zu_brand": np.array([r["brand"] for r in zi if r["role"] == "val_unk"]),
        "zo_brand": np.array([r["brand"] for r in zi if r["role"] == "oe"][::4])}
m = timm.create_model("fastvit_mci0.apple_mclip2_dfndr2b", pretrained=True).to(dev).eval()
def load(p):
    im = Image.open(p).convert("RGB")
    if im.size != (SIZE, SIZE): im = im.resize((SIZE, SIZE), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
@torch.no_grad()
def emb(paths):
    out = []
    for i in range(0, len(paths), 64):
        x = torch.stack([load(p) for p in paths[i:i+64]]).to(dev).float() / 255
        f = F.normalize(m(x), dim=-1) + F.normalize(m(torch.flip(x, [3])), dim=-1)
        out.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(out).numpy()
res = {k: emb(v) for k, v in sets.items()}
# text tower
import open_clip
cm, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b")
tok = open_clip.get_tokenizer("MobileCLIP2-S0")
cm = cm.to(dev).eval()
# check the image towers agree
with torch.no_grad():
    x = torch.stack([load(p) for p in sets["real"][:16]]).to(dev).float() / 255
    a = F.normalize(m(x), dim=-1); b = F.normalize(cm.encode_image(x), dim=-1)
    print("timm vs open_clip image cos:", (a * b).sum(-1).mean().item())
BRANDS = ["Adidas", "Converse", "Nike",
          "Puma", "Reebok", "Vans", "Asics", "Onitsuka Tiger", "New Balance", "Skechers", "Fila", "Under Armour", "Saucony",
          "Brooks", "Mizuno", "Hoka", "On Running", "Salomon", "Lacoste", "Diadora", "Le Coq Sportif", "K-Swiss", "Keds",
          "Crocs", "Timberland", "Dr. Martens", "Merrell", "Li-Ning", "Anta", "Kappa", "Umbro", "Lotto", "Hummel", "Ellesse",
          "Champion", "Supra", "DC Shoes", "Etnies", "Palladium", "Veja", "Golden Goose", "Balenciaga", "Gucci", "Common Projects",
          "Superga", "Bata", "Feiyue", "Warrior", "Karhu", "Saucony", "Altra", "Allbirds", "Steve Madden", "Sperry", "Clarks", "Ecco",
          "Geox", "Kswiss", "Pony", "British Knights", "Lugz", "Heelys", "Peak", "Xtep", "361 Degrees", "Joma", "Hi-Tec"]
TEMPL = ["a photo of a {} sneaker.", "a photo of {} shoes.", "a {} shoe.", "a product photo of a {} sneaker.", "a close-up photo of a {} shoe with its logo."]
EXTRA = ["a photo of a counterfeit fake sneaker.", "a photo of a cheap knockoff sneaker with a fake logo.", "a photo of a generic unbranded sneaker.",
         "a photo of a fake Nike shoe.", "a photo of a fake Adidas shoe.", "a photo of a fake Converse shoe.", "a photo of a sneaker with no logo.",
         "a photo of a canvas sneaker.", "a photo of a running shoe.", "a photo of a basketball shoe.", "a photo of a skate shoe."]
with torch.no_grad():
    T = []
    for b in BRANDS:
        t = F.normalize(cm.encode_text(tok([s.format(b) for s in TEMPL]).to(dev)).float(), dim=-1).mean(0)
        T.append(F.normalize(t, dim=0).cpu())
    E = F.normalize(cm.encode_text(tok(EXTRA).to(dev)).float(), dim=-1).cpu()
res["text_brand"] = torch.stack(T).numpy(); res["text_extra"] = E.numpy()
res["logit_scale"] = np.array(cm.logit_scale.exp().item())
np.savez(f"feats_e0_{SIZE}.npz", brands=np.array(BRANDS), extra=np.array(EXTRA), **res, **meta)
print({k: getattr(v, "shape", None) for k, v in res.items()}, res["logit_scale"])
