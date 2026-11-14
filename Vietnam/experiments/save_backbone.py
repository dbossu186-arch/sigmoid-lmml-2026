"""Save the compressed, traced MobileCLIP2-S0 image backbone (input: [B,3,256,256] float in [0,1]; output: [B,1024]).
Usage: python scripts/save_backbone.py <fp16|int8> <out.pt>"""
import sys, os, torch, timm
from timm.utils.model import reparameterize_model
sys.path.insert(0, os.getcwd())
from osr_export_lib import QuantInner, mci_policy
torch.manual_seed(0); torch.set_num_threads(4)
m = reparameterize_model(timm.create_model("fastvit_mci0.apple_mclip2_dfndr2b", pretrained=True, num_classes=0).eval())
q = QuantInner(m, mci_policy(sys.argv[1])).eval()
with torch.no_grad():
    tr = torch.jit.trace(q, torch.rand(2, 3, 256, 256), check_trace=False)
torch.jit.save(tr, sys.argv[2]); print("saved", sys.argv[2], os.path.getsize(sys.argv[2]) / 1e6, "MB")
