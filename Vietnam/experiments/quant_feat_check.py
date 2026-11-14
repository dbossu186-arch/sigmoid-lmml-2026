"""Backbone feature fidelity of compressed+traced MobileCLIP vs fp32 (Mac, CPU, small).
Usage: python scripts/quant_feat_check.py <timm_model> <int4_min_numel> [n_images]"""
import sys, os, io, csv, numpy as np, torch, timm
from PIL import Image
from timm.utils.model import reparameterize_model
sys.path.insert(0, os.getcwd())
from osr_export_lib import QuantInner, size_policy, mci_policy
torch.set_num_threads(4)
name, thr = sys.argv[1], int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 96
mk = lambda: reparameterize_model(timm.create_model(name, pretrained=True, num_classes=0).eval())
ref = mk()
rows = list(csv.DictReader(open("data/train-output.csv")))[:n]
X = torch.stack([torch.from_numpy(np.asarray(Image.open("data/train-input/" + r["filename"]).convert("RGB")).copy()).permute(2, 0, 1) for r in rows]).float() / 255
for pol_name, pol in [("mci int4+fp16", mci_policy("fp16")), ("mci int4+int8", mci_policy("int8"))]:
    q = QuantInner(mk(), pol).eval()
    with torch.no_grad():
        tr = torch.jit.trace(q, X[:2], check_trace=False)
    buf = io.BytesIO(); torch.jit.save(tr, buf); mb = buf.tell() / 1e6
    with torch.no_grad():
        a = torch.cat([ref(X[i:i + 16]) for i in range(0, n, 16)]); b = torch.cat([tr(X[i:i + 16]) for i in range(0, n, 16)])
    a = torch.nn.functional.normalize(a, dim=1); b = torch.nn.functional.normalize(b, dim=1)
    cs = (a * b).sum(1)
    # nearest-neighbour agreement among the images (proxy for kNN-score stability)
    nn_a = (a @ a.T).fill_diagonal_(-1).argmax(1); nn_b = (b @ b.T).fill_diagonal_(-1).argmax(1)
    print(f"{pol_name:22s} size={mb:.2f}MB cos(fp32,q) mean={cs.mean():.4f} min={cs.min():.4f} NN-agree={(nn_a == nn_b).float().mean():.3f}", flush=True)
