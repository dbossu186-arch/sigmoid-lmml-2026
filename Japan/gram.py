"""Pairwise NCC Gram matrix of PRNU-weighted residuals (a_i = r*I*sqrt(w)), all photos.
usage: gram.py <cache> <out.npz>"""
import sys, glob, os
import numpy as np, torch
cache, out = sys.argv[1:3]
torch.set_num_threads(4)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
files = sorted(glob.glob(f"{cache}/*.npz"))
who = [os.path.basename(f).split("__")[0] for f in files]
names = [os.path.basename(f).split("__")[1][:-4] for f in files]
X = []
for f in files:
    d = np.load(f)
    a = d["r"].astype(np.float32) * d["I"].astype(np.float32) * np.sqrt(d["w"].astype(np.float32))
    # remove global bias pattern & JPEG grid later via mean subtraction across photos
    X.append(torch.from_numpy(a.ravel()).half())
X = torch.stack(X).to(dev).float()
X = X - X.mean(0, keepdim=True)          # remove the pattern common to ALL photos (pipeline artifacts)
X = X - X.mean(1, keepdim=True)
X = X / X.norm(dim=1, keepdim=True)
G = (X @ X.T).cpu().numpy()
np.savez(out, G=G, who=np.array(who), names=np.array(names))
lab = np.array(who); cred = lab != "anonymous"
off = ~np.eye(len(lab), dtype=bool)
same = (lab[:, None] == lab[None]) & cred[:, None] & cred[None] & off
diff = (lab[:, None] != lab[None]) & cred[:, None] & cred[None]
mu, sd = G[diff].mean(), G[diff].std()
print("same z=%.2f same %.5f diff %.5f sd %.5f" % ((G[same].mean() - mu) / sd, G[same].mean(), mu, sd))
ty = lab == "toorisugari"; yk = lab == "yakin"
print("toori x yakin z=%.2f" % ((G[np.ix_(ty, yk)].mean() - mu) / sd))
