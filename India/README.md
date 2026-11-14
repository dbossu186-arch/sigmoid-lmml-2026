# India: Twins Separated at Birth (Sigmoid LMML 2026, Easy)

Find pairs of test images that come from the same source photo, then write `output.csv` (`image_a,image_b`).

## Method (`main.py`)
1. **Decode:** every image is decoded once, in parallel. Each gets a perceptual hash (pHash) and SIFT keypoints.
2. **Candidates:** the 8 nearest neighbours by ResNet-50 embedding (cosine), plus pairs with close pHash. This keeps about 98% of true pairs on train while checking only about 6 candidates per image.
3. **Verification:** SIFT matches plus a RANSAC homography, scored by the share of agreeing keypoints. Then one image is warped onto the other and pixel agreement is measured.
4. **Selection:** each image keeps at most its 2 best partners, since duplicate groups have 2–3 images.

`output.csv` is the submitted prediction file. Run with `python main.py` next to `test-input.zip`; the data is not included.
