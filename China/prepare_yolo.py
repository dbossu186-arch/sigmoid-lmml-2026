"""Convert train_labels.csv into a YOLO detection dataset.

python prepare_yolo.py --data data --out yolo
Classes = every glyph_id in glyphs.csv (incl. 'rare'), so the detector learns
to tell the eight crate glyphs apart from all the look-alikes. A fixed 15%
image split is held out for validation.
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("data"))
ap.add_argument("--out", type=Path, default=Path("yolo"))
a = ap.parse_args()

labels = pd.read_csv(a.data / "train_labels.csv")
glyphs = pd.read_csv(a.data / "glyphs.csv", encoding="utf-8-sig")
classes = list(glyphs.glyph_id)
cid = {g: i for i, g in enumerate(classes)}

imgs = sorted(labels.image.unique())
random.Random(0).shuffle(imgs)
nval = int(0.15 * len(imgs))
split = {im: ("val" if k < nval else "train") for k, im in enumerate(imgs)}

for s in ("train", "val"):
    (a.out / "images" / s).mkdir(parents=True, exist_ok=True)
    (a.out / "labels" / s).mkdir(parents=True, exist_ok=True)

for im, g in labels.groupby("image"):
    src = a.data / "train" / "images" / im
    h, w = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE).shape
    s = split[im]
    dst = a.out / "images" / s / im
    if not dst.exists():
        os.symlink(src.resolve(), dst)
    with open(a.out / "labels" / s / (Path(im).stem + ".txt"), "w") as fh:
        for r in g.itertuples():
            cx = (r.x + r.w / 2) / w
            cy = (r.y + r.h / 2) / h
            fh.write(f"{cid[r.glyph_id]} {cx:.6f} {cy:.6f} {r.w / w:.6f} {r.h / h:.6f}\n")

with open(a.out / "data.yaml", "w", encoding="utf-8") as fh:
    fh.write(f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\n")
    fh.write("names:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(classes)))
print(len(imgs), "images,", nval, "val,", len(classes), "classes")
