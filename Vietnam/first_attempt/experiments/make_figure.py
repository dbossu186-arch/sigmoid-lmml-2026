"""Score-distribution figure: knowns vs unknown proxies, with the chosen threshold.
Usage: python make_figure.py --thr 0.5 --cv runs/cv0 runs/cv1 ... --lobo runs/lobo_a runs/lobo_c runs/lobo_n --out figure.png
Score s = 1 - p(unknown) (higher = more 'known'); reject if s < thr."""
import argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--thr", type=float, required=True)
ap.add_argument("--cv", nargs="+", required=True); ap.add_argument("--lobo", nargs="*", default=[])
ap.add_argument("--out", default="figure.png"); a = ap.parse_args()
def sm(l): e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def sc(l): return 1 - sm(l)[:, -1]
def cat(runs, key): return np.concatenate([sc(np.load(f"{r}/val_logits.npz")[key]) for r in runs])
groups = [("real knowns (5-fold CV held-out)", cat(a.cv, "real_va"), "#2a6fdb"),
          ("Zappos knowns (Nike/adidas/Converse, held-out)", cat(a.cv, "zk_va"), "#7fb2f0"),
          ("Zappos PUMA/Reebok/Vans/ASICS (test brands)", cat(a.cv, "zu_va"), "#d9480f"),
          ("Zappos other unseen brands (OE-val)", cat(a.cv, "oe_va"), "#f59f00")]
if a.lobo: groups.append(("real photos of a held-out known brand (LOBO)", cat(a.lobo, "real_ho"), "#7b2cbf"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
bins = np.linspace(0, 1, 41)
for name, s, c in groups:
    ax[0].hist(s, bins=bins, density=True, histtype="step", lw=2, color=c, label=f"{name} (n={len(s)})")
ax[0].axvline(a.thr, color="k", ls="--", lw=1.5, label=f"threshold = {a.thr}")
ax[0].set_xlabel("s = 1 - p(unknown)   (reject if s < threshold)"); ax[0].set_ylabel("density"); ax[0].set_yscale("log")
ax[0].legend(fontsize=7.5, loc="upper center"); ax[0].set_title("Open-set score distributions")
ts = np.linspace(0.01, 0.99, 99)
for name, s, c in groups:
    frac = [(s >= t).mean() if "known" in name and "unseen" not in name and "LOBO" not in name and "test" not in name else (s < t).mean() for t in ts]
    lab = ("accepted: " if "knowns" in name else "rejected: ") + name.split(" (")[0]
    ax[1].plot(ts, frac, color=c, lw=2, label=lab)
ax[1].axvline(a.thr, color="k", ls="--", lw=1.5); ax[1].axhline(0.85, color="grey", ls=":", lw=1); ax[1].axhline(0.70, color="grey", ls=":", lw=1)
ax[1].set_xlabel("threshold"); ax[1].set_ylabel("fraction"); ax[1].set_ylim(0, 1.01); ax[1].legend(fontsize=7.5, loc="lower left")
ax[1].set_title("Threshold sensitivity (dotted: 0.85 / 0.70 gates)")
plt.tight_layout(); plt.savefig(a.out, dpi=130); print("saved", a.out)
