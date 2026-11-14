# Greece — "Just a siblings' thing" (Medium, 200 pts)

## Storyline
Three oracle models (Ares, Athena, Apollo) classify whether an encounter with a mythological creature is a catastrophic threat needing immediate response. All three report the same test accuracy. For this deployment, missing a true catastrophic threat is far more expensive than a false alarm.
Load the PyTorch weights, run all three on the test set, compute the metrics accuracy hides, decide which oracle guards Delphi.

## Objective
Evaluate the three binary classifiers at decision threshold 0.50. Compare different metrics. Select the best for the stated case (false negatives very costly → recall on positive class matters most).

## Flag
SIGMOID_<WORD>, WORD = uppercase model name (e.g. SIGMOID_ATHENA).

## Input
models/model_ares.pth, model_athena.pth, model_apollo.pth; model.py (shared architecture + feature order); test_dataset.csv; test_labels.csv.
Columns include region, moon phase, creature mass, aggression, divine aura, armor resistance, witness count, omen intensity, oracle-related measurements.
Data: https://drive.google.com/drive/folders/1wg1-xfTWUagA78WRMw3PU9PN4BU1mqns?usp=sharing

## Expected output
The flag + materials (link) proving how it was solved.
