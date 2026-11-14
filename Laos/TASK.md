# LAOS: Bad BOMB (Medium, CV)

Source: https://sigmoid.academy/tasks (popup text copied 2026-11-14)

## Storyline
Decades after historical wars in Southeast Asia, farmers in Laos are still finding old artillery shells buried in their fields. A local charity is using cameras to take close-up pictures of the metal objects they dig up.

Most of the shells are completely empty and safe, but about 2% of them have specific defects showing they might still be live and dangerous. The charity needs to find these dangerous shells fast so they can safely destroy them. But there is a catch: the images are completely unlabeled. You need to build a program that looks at the pictures and flags the odd, defective shells so the safety crew knows exactly which ones to handle with care.

## Objective
You are provided with the images/ dataset folder, in which are pictures with shells. Based on the dataset train a model that correctly identifies 28 defective shells.

## Flag
Format `SIGMOID_<WORD>`. The flag is not hidden in the dataset: the evaluator releases it once the submission passes the scoring criteria.

## Task input
- images/: 1125 images, 512x512 PNG, of missile shells, in different lighting and background settings. Hidden among them are 28 with defects. (Unlabeled.)

## Expected output
Submit a txt file named `test-output.txt` with a list of 28 defective shells. Use `train-output.txt` as the reference for the output format.

## Scoring
- Primary metric: exact-match accuracy on the 28 ground-truth anomalous images: `len(output ∩ ground_truth) / len(ground_truth)`.
- Accuracy ≥ 70% (≥ 20 of 28): 200 points (PASS)
- Accuracy < 70%: 0 points (FAIL)

## Submission
Upload .csv/.txt/.json, max 10 MB. Two hints (locked; penalties; don't unlock without asking).

## Data
Google Drive folder ID `1NY2_0QtTSZC7k439HWP8SUMOrEeFEUow` → `data/`
