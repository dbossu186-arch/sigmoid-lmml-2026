"""Figure: open-set score distributions (knowns vs unknown proxies) and threshold sensitivity.
Input: head_scores.npz written by train.py. Usage: python make_figure.py [head_scores.npz] [figure.png]"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "head_scores.npz"); out = sys.argv[2] if len(sys.argv) > 2 else "figure.png"
thr = float(d["thr"])
G = [("real knowns, 5-fold CV held-out (n=735)", d["oof_s"], "#2a6fdb", "known"),
     ("real knowns in LOBO runs", d["lobo_known"], "#8fb8f0", "known"),
     ("real photos of a held-out known brand (LOBO)", d["lobo_unk"], "#7b2cbf", "unk"),
     ("UT-Zappos50K PUMA/Reebok/Vans/ASICS (test brands)", d["zap_unk"], "#d9480f", "unk"),
     ("UT-Zappos50K Nike/adidas/Converse (catalogue knowns)", d["zap_known"], "#f59f00", "known")]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
bins = np.linspace(-16, 4, 61)
for name, s, c, _ in G:
    ax[0].hist(np.clip(s, -16, 4), bins=bins, density=True, histtype="step", lw=2, color=c, label=f"{name}")
ax[0].axvline(thr, color="k", ls="--", lw=1.5, label=f"threshold = {thr:.2f} (q = 0.06)")
ax[0].set_xlabel("open-set score s = z(class-kNN cosine) + z(max logit)    (reject if s < threshold)")
ax[0].set_ylabel("density"); ax[0].legend(fontsize=7.5, loc="upper left"); ax[0].set_title("Score distributions")
oof, ok = d["oof_s"], d["oof_ok"]
qs = np.linspace(0.0, 0.25, 51)[1:]
ts = [np.quantile(oof, q) for q in qs]
ax[1].plot(qs, [np.mean(ok & (oof >= t)) for t in ts], color="#2a6fdb", lw=2, label="known accuracy (CV, real)")
ax[1].plot(qs, [np.mean(d["zap_unk"] < t) for t in ts], color="#d9480f", lw=2, label="rejection: Zappos test brands")
ax[1].plot(qs, [np.mean(d["lobo_unk"] < t) for t in ts], color="#7b2cbf", lw=2, label="rejection: LOBO real photos (pooled scale)")
ax[1].plot(qs, [np.mean(d["zap_known"] >= t) for t in ts], color="#f59f00", lw=1.5, ls="-.", label="accepted: Zappos catalogue knowns")
ax[1].axvline(0.06, color="k", ls="--", lw=1.5); ax[1].axhline(0.85, color="grey", ls=":"); ax[1].axhline(0.70, color="grey", ls=":")
ax[1].text(0.2, 0.855, "known gate 0.85", fontsize=8, color="grey"); ax[1].text(0.2, 0.705, "rejection gate 0.70", fontsize=8, color="grey")
ax[1].set_xlabel("q = fraction of held-out real knowns rejected (sets the threshold)"); ax[1].set_ylim(0.3, 1.01)
ax[1].legend(fontsize=7.5, loc="lower right"); ax[1].set_title("Threshold sensitivity")
plt.tight_layout(); plt.savefig(out, dpi=130); print("saved", out)
