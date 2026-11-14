# Japan v2: attribution solved, secret not found yet

## Pipeline (run in order, ~3 min total on the dev box)
1. `v2/stage1.py data/dataset work/feats`: wavelet residual (Daniel's `prnu_lib`), caption mask, luminance.
2. `v2/white_only.py work/feats work`: DFT Wiener whitening only. **Do not subtract a common pattern
   in pixel space**: with grain-dominated residuals, every photo leaks into the common estimate at 1/n
   and the subtraction adds a ~-2.5e-3 bias to every pair (5x the camera signal).
3. `v2/center.py work/gram_white.npz work/gram_white_cc.npz`: double-center the off-diagonal of the
   id / rot180 gram matrices (removes the shared component and per-photo offsets exactly).
4. `v2/gattrib.py work/gram_white_cc.npz work/gattrib_white.json 4`: orientation per camera from the
   top eigenvector of (G_id - G_rot), sum-fingerprint scores in gram space, transductive rounds.

## Results
- Credited leave-one-out: 107/108 (first attempt: ~76%).
- Anonymous: Toorisugari 45 = **9 every week**; every other regular 4 per week (week 12: one
  Amayadori/Jihanki coin flip, w12_016). List: `v2/results/toorisugari.txt`, all: `v2/results/attribution.csv`.
- Orientation per photo is unambiguous (best orientation z 6-20, other ~0).

## Ruled out for the secret (all with clean attribution)
- A 2nd camera / hidden subset inside Toorisugari's 63 photos (residual eigen-analysis, block means).
- A component shared by Yakin and Toorisugari (same phone model): cross block equals every other pair.
- Image-frame watermark on Toorisugari's posts; extra component in their anonymous vs credited photos.
- Chroma residuals (no camera structure at all: PRNU and grain are monochrome).
- Letter-shaped structure in fingerprint energy maps, local agreement maps (all, per week), local
  correlation (tampering) maps, image-frame averages.
- Shifted copies of fingerprints (full cross-correlation scan; off-zero peaks are JPEG-grid, multiples of 8).
- JPEG structure: all files identical PIL JFIF, no trailing data, no comments.
- Dark-on-black content in the caption bars (bars are solid 0).
- Orientation bits of Toorisugari's photos as 5-/8-bit text (no readable decode).

## Not yet tried
- Reading the captions of Toorisugari's 45 photos (acrostic / repeated phrase).
- Effect parameters (grain level, warmth, caption style/position) of the extra 5 per week.
- Scene content of Toorisugari's photos (they imitate the other regulars' themes: vending machines,
  night streets, rain, trams).
