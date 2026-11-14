# BHUTAN - The 300-Year Old Tiger Nest Painting (Medium, Computer Vision)

## Storyline (summary)
A ~300-year-old painted scroll of Paro Taktsang (Tiger's Nest) — monks at the cliff-edge viewpoint, a border strip of hand-lettered verse, a red seal in a corner — was cut by a disgraced archive monk into vertical strips, shuffled and misfiled. The scroll's border hides the flag of Bhutan. Write code that undoes the shredding.

## Objective
Train set: vertical strips of the original image (shuffled), plus train output with the correct order and the reconstructed image. Reconstruct the original strip order and the image using CV techniques; save the order.

## Flag
Format SIGMOID_<WORD>. Not stated in plain text anywhere in the document/dataset. Released automatically by the evaluator once main.py properly reconstructs the hidden test PNG image and test strips order JSON (passing scoring criteria).

## Input
- train-input/strip_001.png ... strip_192.png (192 strips). Filenames are randomized; carry no positional info. All strips same width/height, perfectly cut vertically.
- train-output/reconstructed_image.png (the original image), train-output/reconstructed_order.json (order of strips).
- IMPORTANT: main.py must read input from a generic folder named `input` (not train-/test-).

## Pinned environment
Python 3.13, numpy==2.5.3, opencv-python==5.0.0.93, scipy==1.18.1, scikit-image==0.26.0 (requirements.txt).

## Expected output (upload: main.py, .py only, max 10 MB)
- main.py ("reassembly" code)
- reconstructed_image.png saved in the same dir as main.py
- reconstructed_order.json in the same dir as main.py, format: ["strip_047.png", "strip_012.png", ...]
- The flag string returned by the evaluator after submission.

## Scoring
Evaluated on different (hidden) strips/image.
- SSIM between ground-truth image and reconstructed_image.png >= 0.90
- Order match vs ground-truth order >= 0.90 (full left-right mirror accepted as equivalent)
- Hard timeout 20 s for main.py (blocks brute force)
All three met: 200 pts (PASS, flag released). Any missed: 0.
Hints 1-3 available (penalty) — do not unlock.

## Data
https://drive.google.com/drive/folders/1L4L9B9ui35VR5XzGGiv2yBTyqltQCY4D
