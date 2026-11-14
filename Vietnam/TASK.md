# VIETNAM: That isn't my shoe! (Medium, CV)

Source: https://sigmoid.academy/tasks (popup text copied 2026-11-14)

## Storyline
Vietnam is one of the world's largest athletic-footwear manufacturing hubs: Nike, Adidas and Converse all run major production there. On the night shift at a footwear plant outside Hanoi, a scanner named Sherlock Soles sorts every sneaker into three bins, Nike, Adidas and Converse, until strangers start arriving. A plain classifier is forced to guess and cheerfully drops a Croc into "Adidas." So Sherlock learns to trust only what it can verify. A genuine shoe always carries the real thing, the exact stitch tolerances and sole geometry an authentic Nike never gets wrong, and earns its label. Everything else falls into one honest bucket: unknown. The counterfeit Nike that sits right atop the real ones but drifts on those tiny tells is unknown. The knockoff maker who copies a rival's whole form factor, a shoe shaped like a Converse but honest in none of the details, is unknown. Only the label makes it a label; the rest, however convincing, stays unknown.

## Objective
Open-set recognition. Train only on the three known brands, then label each test image as one of them or reject it as the literal `unknown`. The test set mixes knowns with unseen look-alike sneakers (Puma, Reebok, Vans, Asics).

## Flag
`SIGMOID_<WORD>`, released by the evaluator only when both scoring gates pass.

## Task input
- Real 256×256 sneaker photos. train-input/ + train-output.csv: 735 known images (nike/adidas/converse).
- The test set is held by the evaluator and is not provided. Images are referenced by filename only.
- Environment for testing: Python 3.10–3.12 with **torch==2.11.0, torchvision==0.26.0, numpy==2.4.6, Pillow==12.3.0**.

## Expected output
Submit the trained model as a **single .pt checkpoint** (no code needed; the platform runs it). Max file size **10 MB**. The exact checkpoint format / interface the platform expects must be read from the data folder (README or example files); check before training.

## Scoring
Two metrics on the test set:
- Known accuracy: share of known images labelled with the correct brand.
- Rejection rate: share of unknown images labelled `unknown`.
- PASS (200 points, flag): known accuracy ≥ 0.85 AND rejection rate ≥ 0.70.
- FAIL (0): either metric below its threshold.

## Submission
Upload .pt, max 10 MB. Three hints (locked; don't unlock without asking).

## Data
Google Drive folder ID `12INJs3C0w8eWLOWy43rP74HhHBLghhbF` → `data/`
