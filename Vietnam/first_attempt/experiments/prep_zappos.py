"""Build an index of UT-Zappos50K 'Sneakers and Athletic Shoes' images.
Roles:
  val_unk  : PUMA / Reebok / Vans / ASICS (+ Onitsuka Tiger) -> the test's unknown brands, VALIDATION ONLY
  val_known: Nike / adidas / Converse (all sub-lines)           -> domain-shifted known proxy (split half train-aux / half val)
  oe       : every other brand                                  -> outlier exposure (split 80% train / 20% val by brand)
Writes ext/zappos_index.csv with columns path,brand,role,label,split
"""
import os, csv, random, sys
root = sys.argv[1] if len(sys.argv) > 1 else "ext/ut-zap50k-images-square/Shoes/Sneakers and Athletic Shoes"
TEST_UNK = ["puma", "reebok", "vans", "asics", "onitsuka"]
KNOWN = {"nike": "nike", "adidas": "adidas", "converse": "converse"}
rng = random.Random(0)
rows = []
brands = sorted(os.listdir(root))
oe_brands = [b for b in brands if not any(k in b.lower() for k in TEST_UNK + list(KNOWN))]
rng.shuffle(oe_brands)
oe_val_brands = set(oe_brands[: len(oe_brands) // 5])
for b in brands:
    bl = b.lower()
    files = sorted(f for f in os.listdir(os.path.join(root, b)) if f.endswith(".jpg"))
    if any(k in bl for k in TEST_UNK):
        role, label = "val_unk", "unknown"
        split = ["val"] * len(files)
    elif any(k in bl for k in KNOWN):
        label = [v for k, v in KNOWN.items() if k in bl][0]
        role = "val_known"
        # product-level split (filename = productid.colorid.jpg) -> keep colors of a product together
        split = ["val" if (hash(f.split('.')[0]) % 2 if False else int(f.split('.')[0]) % 2) else "train" for f in files]
    else:
        role, label = "oe", "unknown"
        split = ["val" if b in oe_val_brands else "train"] * len(files)
    for f, s in zip(files, split):
        rows.append((os.path.join(root, b, f), b, role, label, s))
with open("ext/zappos_index.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "brand", "role", "label", "split"]); w.writerows(rows)
from collections import Counter
print(Counter((r[2], r[3], r[4]) for r in rows))
