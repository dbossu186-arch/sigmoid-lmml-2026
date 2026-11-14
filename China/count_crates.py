"""Count each crate's glyph in every set with the trained detector.

python count_crates.py --weights runs/detect/glyphs/weights/best.pt --data data --conf 0.25
Writes counts.csv (crate, set, glyph_id, count), detections_crates.csv (every
detection of any class, for later re-thresholding) and prints the per-set
counts at several confidence thresholds so the decoding can be checked for
stability.
"""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data", type=Path, default=Path("data"))
ap.add_argument("--imgsz", type=int, default=896)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--out", type=Path, default=Path("counts.csv"))
ap.add_argument("--device", default=None)
a = ap.parse_args()

model = YOLO(a.weights)
names = model.names
crates = pd.read_csv(a.data / "crates.csv", encoding="utf-8-sig")

det = []
for crate in crates.crate:
    for set_dir in sorted((a.data / "crates" / crate).iterdir()):
        files = sorted(set_dir.glob("*.jpg"))
        for f in files:
            # one image per call: batching 30 rubbings tripped Ultralytics' NMS
            # time limit and silently dropped whole batches of detections
            r = model.predict(str(f), imgsz=a.imgsz, conf=0.05, iou=0.5, verbose=False,
                              max_det=300, device=a.device)[0]
            for b, c, s in zip(r.boxes.xywh.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                det.append([crate, set_dir.name, f.name, names[int(c)], s, *b])
det = pd.DataFrame(det, columns=["crate", "set", "image", "glyph_id", "conf", "cx", "cy", "w", "h"])
det.to_csv("detections_crates.csv", index=False)

rows = []
for crate, g in zip(crates.crate, crates.glyph_id):
    for s in sorted(det[det.crate == crate].set.unique()):
        d = det[(det.crate == crate) & (det.set == s) & (det.glyph_id == g)]
        rows.append([crate, s, g, int((d.conf >= a.conf).sum())]
                    + [int((d.conf >= t).sum()) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])
out = pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count", "c10", "c20", "c30", "c40", "c50", "c60"])
out[["crate", "set", "glyph_id", "count"]].to_csv(a.out, index=False)
print(out.to_string(index=False))
