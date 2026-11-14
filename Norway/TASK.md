# NORWAY: The Nisse's Bad Shift (Medium)

Source: https://sigmoid.academy/tasks (popup text copied 2026-11-14)

## Storyline
When Norway's municipal registry digitized its archives, the night shifts were left to the office nisse, a small, tireless caretaker spirit that works flawlessly as long as it gets its porridge, with a pat of butter on top.

Every now and then, somebody forgot the butter. The nisse never stopped working, but out of quiet spite it would spend that whole night swapping two digits every single time it wrote them down. The next evening the porridge came back, and so did perfect work. Nobody noticed. Find the nisse's bad nights, put the digits back, and give the registry a classifier it can trust again.

## Objective
Some labels in a handwritten-digit dataset are wrong, not at random, but because of the nisse's bad nights. Find the mislabeled samples in `test-input.csv`, correct them, and retrain a classifier on your corrected labels.

## Flag
`SIGMOID_<WORD>`, released automatically when the submission reaches the passing score.

## Task input
| File | Rows | Columns |
|---|---|---|
| train-input.csv | 1,198 | sample_id, pixel_0 … pixel_63, given_label |
| train-output.csv | 1,198 | sample_id, corrected_label (the true label) |
| test-input.csv | 599 | same as train-input.csv |

- pixel_0 … pixel_63: a flattened 8×8 grayscale scan, values 0–16.
- given_label: the label on file, possibly wrong.
- **test-input.csv comes from other nights of the same registry. Not every bad night goes wrong the same way.**
- Python 3.11; see requirements.txt for pinned libraries.
- Scans come from UCI "Optical Recognition of Handwritten Digits" (Alpaydin & Kaynak, 1998) via `sklearn.datasets.load_digits()` (BSD-3-Clause).

## Expected output
1. `test-output.csv` (scored automatically): exactly two columns, one row for every sample_id in test-input.csv:
   ```
   sample_id,corrected_label
   IMG_XXXX,<digit 0-9>
   ```
2. The retrained classifier (e.g. `model.joblib`), trained on the corrected labels. It is reviewed but not scored.

## Scoring
Scoring is on the label repairs, not on any model's accuracy. A repair counts only if a mislabeled sample gets its true label; changing a correct label counts against you. Score = F1 (harmonic mean of precision and recall) of the repairs.
- F1 ≥ 0.93: 200 points (PASS)
- F1 < 0.93: 0 points (FAIL)

## Submission
Upload, accepted types .csv / .txt / .json, max 10 MB. Two hints exist (locked, they carry penalties; don't unlock them without asking).

## Data
Google Drive folder ID `1wNDRDlEVopfH3p72PN973rtvle6-Gbo9` → `data/`
