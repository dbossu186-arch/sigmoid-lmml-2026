import sys, numpy as np
from sklearn.metrics import roc_auc_score
d = np.load(sys.argv[1]); B = list(d["brands"]); ls = float(d["logit_scale"])
X, y, ZK, zky, ZU, ZO = d["real"], d["real_y"], d["zk"], d["zk_y"], d["zu"], d["zo"]
zub = np.array([b.split()[0].upper() for b in d["zu_brand"]])
TB, TE = d["text_brand"], d["text_extra"]
for name, drop, ext in [("all", (), True), ("brands", (), False), ("no-test-brands", ("Puma","Reebok","Vans","Asics","Onitsuka Tiger"), True)]:
    N = np.concatenate([np.stack([TB[i] for i in range(3, len(B)) if B[i] not in drop])] + ([TE] if ext else []))
    def sc(F_): K = F_ @ TB[:3].T; return K.max(1) - (F_ @ N.T).max(1), K.argmax(1)
    sr, pr = sc(X); sk, pk = sc(ZK); su, _ = sc(ZU); so, _ = sc(ZO)
    print(f"== {name}: AUC real-zu={roc_auc_score(np.r_[np.ones(len(sr)),np.zeros(len(su))],np.r_[sr,su]):.4f} zk-zu={roc_auc_score(np.r_[np.ones(len(sk)),np.zeros(len(su))],np.r_[sk,su]):.4f}")
    for q in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]:
        t = np.quantile(sr, q) if q > 0 else 0.0
        print(f"  t={t:+.4f} (real q={q}) knownacc real={np.mean((pr==y)&(sr>=t)):.3f} zk={np.mean((pk==zky)&(sk>=t)):.3f} | rej zu={np.mean(su<t):.3f} "
              + " ".join(f"{b}={np.mean(su[zub==b]<t):.2f}" for b in sorted(set(zub))) + f" | zo={np.mean(so<t):.3f}")
# what do misses look like: top neg brand for real images that are rejected at t=0
N = np.concatenate([TB[3:], TE]); names = B[3:] + list(d["extra"])
sr = (X @ TB[:3].T).max(1) - (X @ N.T).max(1); idx = np.argsort(sr)[:12]
print("hardest real knowns:", [(int(i), ["adidas","converse","nike"][y[i]], names[(X[i] @ N.T).argmax()], round(float(sr[i]),3)) for i in idx])
