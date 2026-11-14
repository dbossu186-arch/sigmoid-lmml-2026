# LAOS "Bad BOMB": unsupervised shell defect detection

Output: `test-output.txt` lists 28 images as `images/image_XXXX.png`, one per line. It uses CRLF line breaks and has no trailing newline, the same as `data/train-output.txt`.

## Data findings
- The Drive folder held 1241 files. There are 1225 unique names, `image_0001.png` to `image_1225.png`, with no gaps. The other 16 files are the `__dup_` copies of `image_0001` to `image_0016`, and all 16 are byte-identical to their originals. No two of the 1225 originals have the same content (checked with md5). So the "1125" in the task text is not explained by duplicates. It is most likely a typo for 1225. The pipeline reads only `image_????.png` names.
- `train-output.txt` has 28 lines like `images/image_0412.png` and shows the format only. 27 of the 28 names (for example 2891, 2764) do not exist in the dataset. The one that does, `image_0619.png`, is a normal shell: it ranks 715 of 1225, and I checked it by eye. So the file contains no usable labels.
- The images are Blender renders of one shell model with 2 backgrounds (light diamond plate and rusty plate), varying lighting, and small shifts, tilts and scale changes. The PNG metadata is the same for every image except render date and time, and it carries no signal.

## Method (`solve.py`)
1. **Canonical crop.** Segment the shell with a loose "green" colour mask. Keep the largest component plus the base piece below the grey driving band, and take the convex hull. Rotate the long axis to vertical (PCA), crop tightly, and resize to 224x672. After this step, a pixel position is the same part of the shell in every image.
2. **Four detectors.** Each gives a robust z-score per image (median/MAD):
   - `pix`: standardise colour per image inside the shell, take the per-pixel robust z against the median shell, smooth with a 5x5 window, and use the max.
   - `wrn`, `dinos`, `dinob`: patch features from WideResNet50-2 (layer2+layer3), DINOv2 ViT-S/14 and DINOv2 ViT-B/14, on a 42x14 grid. A position-aware kNN (PatchCore/PaDiM hybrid) compares each patch only with patches within ±1 cell in all *other* images. The patch score is the mean distance to the 5 nearest neighbours; k=5 keeps the ~2% of defective images in the bank from hiding each other. Only shell patches count. The image score is the mean of the top 3 patches.
3. **Ensemble.** Average the four z-scores after dropping each image's largest one (trimmed mean), so an image must be flagged by at least 3 of the 4 detectors. This removes single-detector outliers. One example is `image_1067`, a closer camera shot that only DINOv2-B flags.
4. **Visual review swap** (`REVIEW_SWAPS`; turn it off with `--no_review`). This swap changes 1 of the 28 names. See below.

Run on the GPU machine: `python solve.py --crops_dir cj` (crops exported with `--export_crops cj`), or `python solve.py --data data/images`. To re-rank and redraw the figure from the cached scores: `python solve.py --from_scores --work work --crops_dir <crops> --figure figure.png`.

## Evidence
I inspected the top 60 by eye. The defects come in four repeated families:

| family | images | n |
|---|---|---|
| A: notch/dent on the ogive-body seam plus a rust streak above the band | 0215 0904 0781 0843 1158 1170 0402 0623 0506 0763 | 10 |
| B: rust streak running down from the tip, plus a lower streak | 0471 0800 0466 0935 1195 | 5 |
| C: hole just above the driving band, with a deformed band | 0832 0736 0965 1071 0160 0613 0520 0248 0687 0878 0955 | 11 |
| D: chipped, jagged tip plus scratched/pitted body | 0251 0269 0408 | 3 |

That makes 29 visually defective images for 28 slots. The trimmed-mean top 28 is exactly these 29 minus `0955` (rank 30). `0955` is an unmistakable family-C hole. `0408` is the weakest-looking family-D member. So the final list swaps `0408` out and `0955` in. Every submitted image shows a visible defect.

**Score gap** (`figure.png`, left). Rank 28 scores 2.36 and the first non-defect (`0842`, rank 29) scores 2.03. Below that the curve is a smooth tail. The histogram shows the 28 picks separated from the bulk of normal images. Ranks 29 to 60 look like normal shells. The right side of the figure shows the 28 submitted canonical crops.

**Detector agreement.** 26 of the 28 are in the top 28 of the plain mean ensemble too. `pix` alone misses family D, which is subtle in colour, and DINOv2 finds it.

## Ambiguities
- One of the 29 visually defective images is not in the ground truth. My guess is a family-D image (`0408`), but it could be any of them. Expected score: 27 or 28 out of 28, with a pass needing 20.
- The task says 1125 images but there are 1225 unique ones.

## Files
`solve.py`, `test-output.txt`, `figure.png`, `work/scores.csv` (per-detector z-scores and the ensemble for all 1225 images), `work/pmap_dinos.npy` and `work/pmap_dinob.npy` (patch anomaly maps).

## Graded result
Passed on the platform: 21 of 28 defective shells matched (accuracy 0.75, threshold 0.70). The visual self-check overestimated confidence (expected 27–28).

The image data belongs to the organisers and is not included; put it in `data/images/` to rerun `solve.py`.
