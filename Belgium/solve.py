"""Fry-ndship (Sigmoid, Belgium): find the word Mariette observed.

Every length column is the field's rectangle perimeter 2(L + A/L) times a constant,
plus a planted residual that spells a word when plotted against the fitted value:

  ditch_m  (farmer's declaration)   = 0.50 * perimeter + REGRESSION
  hedge_m  (noted by Mariette)      = 0.04 * perimeter + SUPERVISION
  edge_m   (measured by Mariette)   = 0.41 * perimeter + ditch + hedge + PREDICTION

So the analyst's edge model gets R^2 > 0.99 while the words hide in the residuals.
The observation that belongs to Mariette, her own on-site hedge note, spells
SUPERVISION.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "task_belgium/potato_fields.csv"
d = pd.read_csv(path)
d["perimeter_m"] = 2 * (d.long_side_m + d.area_ha * 1e4 / d.long_side_m)

fits = [
    ("ditch_m", ["perimeter_m"], "ditch (farmer's declaration)"),
    ("hedge_m", ["perimeter_m"], "hedge (noted by Mariette on site)"),
    ("edge_m", ["perimeter_m", "ditch_m", "hedge_m"], "edge (measured by Mariette)"),
]

fig, axes = plt.subplots(len(fits), 1, figsize=(18, 4 * len(fits)))
for ax, (y, cols, label) in zip(axes, fits):
    X = np.column_stack([np.ones(len(d))] + [d[c] for c in cols])
    beta, *_ = np.linalg.lstsq(X, d[y], rcond=None)
    fitted = X @ beta
    resid = d[y] - fitted
    r2 = 1 - resid.var() / d[y].var()
    print(f"{y} ~ {' + '.join(cols):35s} coef={np.round(beta[1:], 4)}  R2={r2:.6f}")
    ax.scatter(fitted, resid, s=2, c="k")
    ax.set_title(f"{label}: {y} ~ {' + '.join(cols)}   (R² = {r2:.5f})")
    ax.set_xlabel(f"fitted {y}")
    ax.set_ylabel("residual (m)")

plt.tight_layout()
plt.savefig("residuals.png", dpi=110)
print("Mariette's hedge observations spell SUPERVISION -> SIGMOID_<redacted>")
