"""Turn counts.csv into the word and draw the evidence figure.

python decode.py [counts.csv]
Each crate's five per-set counts are bimodal: either "the glyph is all over
this bundle" (~8-20 detections in 30 rubbings) or "background rate" (0-4).
High = 1, low = 0, set_1 is the most significant bit; the 5-bit number is a
1-based alphabet index (1 = A).
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

counts = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "counts.csv")
crates = sorted(counts.crate.unique())
M = np.array([[int(counts[(counts.crate == c) & (counts.set == f"set_{k}")]["count"].iloc[0])
               for k in range(1, 6)] for c in crates])

# threshold: largest gap in the sorted counts (the bimodal split)
v = np.sort(M.ravel())
gap = int(np.argmax(np.diff(v)))
thr = (v[gap] + v[gap + 1]) / 2
bits = (M > thr).astype(int)
idx = bits @ (2 ** np.arange(4, -1, -1))
word = "".join(chr(64 + i) for i in idx)

print(f"threshold {thr:.1f} (low max {v[gap]}, high min {v[gap + 1]})")
for c, m, b, i in zip(crates, M, bits, idx):
    print(c, list(m), "".join(map(str, b)), i, chr(64 + i))
print("WORD:", word)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.imshow(M, cmap="Greys", vmin=0, vmax=M.max() * 1.15)
for r in range(M.shape[0]):
    for k in range(5):
        ax.text(k, r, f"{M[r, k]}", ha="center", va="center",
                color="white" if bits[r, k] else "black", fontsize=11)
    ax.text(5.0, r, f"{''.join(map(str, bits[r]))} = {idx[r]:2d} -> {chr(64 + idx[r])}",
            va="center", fontsize=11, family="monospace")
glyph = dict(zip(counts.crate, counts.glyph_id))
ax.set_yticks(range(len(crates)), [f"{c} ({glyph[c]})" for c in crates])
ax.set_xticks(range(5), [f"set_{k}" for k in range(1, 6)])
ax.set_xlim(-0.5, 7.2)
ax.set_title(f"Detections of each crate's glyph per set (threshold {thr:.1f})\nhigh=1, set_1 = MSB, A=1  ->  {word}")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig("figure.png", dpi=150)
