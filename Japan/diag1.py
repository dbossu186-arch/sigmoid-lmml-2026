import sys, json, numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out=sys.argv[1]
d=json.load(open(f"{out}/stage2.json")); C=np.load(f"{out}/pair_ncc.npy")
own=np.array(d["owners"]); people=sorted(set(own)-{"anon"})
cred=own!="anon"
for p in people:
    i=np.where(own==p)[0]
    same=C[np.ix_(i,i)][~np.eye(len(i),dtype=bool)].mean()
    other=C[np.ix_(i,np.where(cred&(own!=p))[0])].mean()
    print(p,"same %.4f other %.4f"%(same,other), " vs each:", {q: round(float(C[np.ix_(i,np.where(own==q)[0])].mean()),4) for q in people})
fig,ax=plt.subplots(2,3,figsize=(18,10))
for a,p in zip(ax.flat,people):
    K=np.load(f"{out}/K_{p}.npy"); s=ndimage.gaussian_filter(K,3)
    a.imshow(s,cmap="gray",vmin=np.percentile(s,1),vmax=np.percentile(s,99)); a.set_title(p)
plt.savefig(f"{out}/K_smooth.png",dpi=70)
fig,ax=plt.subplots(2,3,figsize=(18,10))
for a,p in zip(ax.flat,people):
    K=np.load(f"{out}/K_{p}.npy")
    a.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(K)))+1e-3),cmap="magma"); a.set_title(p)
plt.savefig(f"{out}/K_fft.png",dpi=70)
