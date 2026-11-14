"""Train the glyph detector (run on the GPU box).

python train_yolo.py --yaml yolo/data.yaml --epochs 120
No horizontal flips: oracle-bone glyphs are written in either orientation,
but a mirrored glyph is still the same glyph, so fliplr is kept at 0.5 only
if --fliplr is passed (default off, to stay faithful to the labels).
"""
import argparse
import multiprocessing as mp
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # shared GPU box: keep CPU use small

import torch

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--yaml", default="yolo/data.yaml")
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--imgsz", type=int, default=896)
ap.add_argument("--fliplr", type=float, default=0.0)
ap.add_argument("--name", default="glyphs")
a = ap.parse_args()

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)  # py3.14 forkserver breaks dataloader workers
torch.set_num_threads(2)
model = YOLO(a.model)
model.train(
    data=a.yaml, epochs=a.epochs, imgsz=a.imgsz, batch=8, device=0, workers=2,
    fliplr=a.fliplr, mosaic=1.0, hsv_h=0, hsv_s=0, degrees=5, scale=0.4,
    patience=40, name=a.name, exist_ok=True, seed=0, plots=True,
)
