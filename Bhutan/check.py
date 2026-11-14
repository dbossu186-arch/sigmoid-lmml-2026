"""Local stand-in for the evaluator.

Usage: python check.py [strips_dir gt_image gt_order_json]
Copies main.py into a fresh temp dir with the strips under input/, runs it with
a 20 s timeout, then scores:
  * SSIM(gt image, reconstructed_image.png) >= 0.90  (grayscale, skimage)
  * order match >= 0.90 (position-wise, mirror accepted; also reports the
    fraction of correct adjacent pairs, a more lenient metric)
With --synthetic N it also shreds random crops of the train image (and its
horizontally/vertically flipped versions) with fresh random names to make
held-out cases.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

HERE = Path(__file__).resolve().parent


def score(order, gt_order, img, gt_img):
    pos = np.mean([a == b for a, b in zip(order, gt_order)])
    pos_m = np.mean([a == b for a, b in zip(order[::-1], gt_order)])
    gt_pairs = set(zip(gt_order, gt_order[1:]))
    adj = np.mean([(a, b) in gt_pairs or (b, a) in gt_pairs for a, b in zip(order, order[1:])])
    if img.shape != gt_img.shape:
        s = 0.0
    else:
        g1 = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = max(ssim(g1, g2, data_range=255), ssim(g1, g2[:, ::-1], data_range=255))
    return dict(order=max(pos, pos_m), adjacency=adj, ssim=s)


def run_case(strips: dict[str, np.ndarray], gt_order, gt_img):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(HERE / "main.py", td / "main.py")
        (td / "input").mkdir()
        for n, im in strips.items():
            cv2.imwrite(str(td / "input" / n), im)
        t = time.time()
        r = subprocess.run([sys.executable, "main.py"], cwd=td, capture_output=True, text=True, timeout=20)
        dt = time.time() - t
        if r.returncode:
            print(r.stderr)
            return None
        order = json.load(open(td / "reconstructed_order.json"))
        img = cv2.imread(str(td / "reconstructed_image.png"))
        res = score(order, gt_order, img, gt_img)
        res["time"] = dt
        res["pass"] = res["ssim"] >= 0.9 and res["order"] >= 0.9 and dt < 20
        return res


def shred(img, w, rng):
    n = img.shape[1] // w
    img = img[:, : n * w]
    names = [f"strip_{k:03d}.png" for k in range(1, n + 1)]
    perm = rng.permutation(n)
    strips, gt = {}, []
    # strip at original position p gets name names[perm[p]]
    for p in range(n):
        nm = names[perm[p]]
        strips[nm] = img[:, p * w:(p + 1) * w]
        gt.append(nm)
    return strips, gt, img


def main():
    d = HERE / "data"
    gt_img = cv2.imread(str(d / "train-output/reconstructed_image.png"))
    gt_order = json.load(open(d / "train-output/reconstructed_order.json"))
    strips = {p.name: cv2.imread(str(p)) for p in (d / "train-input").glob("*.png")}
    print("train:", run_case(strips, gt_order, gt_img))

    rng = np.random.default_rng(0)
    variants = [gt_img, gt_img[:, ::-1], gt_img[::-1], cv2.rotate(gt_img, cv2.ROTATE_180)]
    allpass = True
    for k in range(int(sys.argv[1]) if len(sys.argv) > 1 else 8):
        base = variants[k % 4]
        H, W = base.shape[:2]
        h = int(rng.integers(H // 2, H + 1)); wdt = int(rng.integers(W // 2, W + 1))
        y = int(rng.integers(0, H - h + 1)); x = int(rng.integers(0, W - wdt + 1))
        crop = np.ascontiguousarray(base[y:y + h, x:x + wdt])
        sw = int(rng.choice([4, 6, 8, 10, 12, 16]))
        st, gt, im = shred(crop, sw, rng)
        res = run_case(st, gt, im)
        allpass &= bool(res and res["pass"])
        print(f"synthetic {k}: {crop.shape[1]}x{crop.shape[0]} w={sw} n={len(st)} ->", res)
    print("ALL SYNTHETIC PASS" if allpass else "SOME SYNTHETIC FAIL")


if __name__ == "__main__":
    main()
