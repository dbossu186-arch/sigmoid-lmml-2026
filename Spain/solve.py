"""¡Feliz Navidad! (Sigmoid, Spain): rebuild the office's Christmas board.

1. Price per unit = Total_paid_eur / Quantity.
2. Office rule: keep a receipt only if Fair_min_eur <= price per unit <= Fair_max_eur
   (cheaper = fake, dearer = tourist trap). 77 of 577 receipts survive.
3. Board: one column per pilot (Pilot_no 1-21), one row per item on the office
   list (List_no 1-6, top to bottom). A cell is filled if that pilot brought that
   souvenir at a fair price.

The board reads  c R u C i O  ->  SIGMOID_<redacted>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = sys.argv[1] if len(sys.argv) > 1 else "task_spain"
guide = pd.read_csv(f"{folder}/office_price_guide.csv")
p = pd.read_csv(f"{folder}/purchases.csv").merge(guide, on="Product")
p["price_per_unit"] = p.Total_paid_eur / p.Quantity
fair = p[(p.price_per_unit >= p.Fair_min_eur) & (p.price_per_unit <= p.Fair_max_eur)]
print(f"{len(fair)} of {len(p)} receipts at a fair price")

board = np.zeros((6, 21), dtype=int)
for _, r in fair.iterrows():
    board[r.List_no - 1, r.Pilot_no - 1] = 1

for row in board:
    print("".join("#" if v else "." for v in row))

labels = guide.sort_values("List_no")
fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(board, cmap="Greens", vmin=0, vmax=1.3)
ax.set_xticks(range(21), [str(i) for i in range(1, 22)])
ax.set_yticks(range(6), [f"{n}. {name}" for n, name in zip(labels.List_no, labels.Product)])
ax.set_xticks(np.arange(-0.5, 21), minor=True)
ax.set_yticks(np.arange(-0.5, 6), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("Pilot_no")
ax.set_title("Christmas board: fair-price souvenirs per pilot  ->  c R u C i O  (SIGMOID_<redacted>)")
plt.tight_layout()
plt.savefig("board.png", dpi=120)
