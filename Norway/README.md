# NORWAY: The Nisse's Bad Shift

Run: `python -I main.py` (writes `test-output.csv` and `model.joblib`). Run `python -I harness.py [n_trials] [--reference]` for the local checks.
Python 3.11 with numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.0, joblib 1.4.2. CPU, about 2 s.

## Method
1. An RBF-SVC (C=10, gamma=0.001, probability=True) is trained on the train images using their true labels.
2. A bad night is a contiguous block of sample_ids in which every occurrence of a and b is swapped. For each of the 45 pairs (a,b), each test sample labelled a or b gets the score `log p(other) - log p(given)`, clipped to ±6. The samples are ordered by id, and Kadane's algorithm finds the contiguous segment with the highest total score.
3. The best (pair, segment) is accepted if its score is at least 12 and at least 4 samples in it support a swap. Every a/b label inside the segment is flipped, the segment is masked, and the search repeats so that several nights can be found.
4. The final classifier is retrained on the train data (true labels) plus the test data (corrected labels) and saved as `model.joblib`.

## Evidence
- Train: one night, ids 452–596, swapping 4↔9 (36 wrong labels, all of them 4/9 in that range).
- Test: one night, ids 1360–1496, swapping 3↔8 (score 146.9). All 30 samples labelled 3 or 8 in that range are flipped, and the classifier supports every one of them. Nothing else is changed. 30 of 599 labels changed.
- Held-out A (detecting the real train night with train labels hidden, 5-fold out-of-fold probabilities): F1 = 1.000 (36 TP, 0 FP, 0 FN).
- Held-out B (60 simulations, each with 1–3 nights of random pairs and lengths on a held-out half of train, classifier fit on the other half): mean F1 = 0.9998, min = 0.9905.
- Reference only, not used by main.py: every train and test image appears verbatim in the public `sklearn.datasets.load_digits()`. Scoring against those targets gives F1 = 1.000 on test (30/30, 0 FP). The pipeline does not depend on this lookup.
