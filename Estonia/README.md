# Estonia: "This Data Is Perfect!"

**True accuracy: 1511/1971 = 0.766616 (76.66%). Flag: `SIGMOID_77`** (76.66% rounded to a whole percent; truncated would be 76).

Reproduce with `python solve.py` (pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.1, Python 3.12). The shipped `data/project_pipeline.py` prints `Test accuracy: 0.9342` exactly.

## The three questionable choices

| # | Choice in `project_pipeline.py` | Verdict |
|---|---|---|
| 1 | `StandardScaler().fit_transform(X)` runs on the full dataset **before** the split, so test rows contribute to the mean and standard deviation. | **Decoy.** It is textbook leakage, but scaling only uses 6 means and 6 standard deviations, and with 1971 rows a 20% test share barely moves them. |
| 2 | `train_test_split` is neither stratified nor balanced, on imbalanced labels (29.1% fraud) scored with plain accuracy. | **Decoy.** Stratifying does not change the score. |
| 3 | Identity fields are dropped ("never used for modeling") and rows are split **at random**. The split ignores who submitted each application. | **Real bug.** |

### Why #3 inflates the score
- 1971 applications come from only **900 people**, where a person is (`date_of_birth`, `citizenship`): 360 people applied once, 195 twice, 159 three times, 186 four times.
- The label is the same across all of a person's resubmissions, and their features are near copies (for example, four FR 1960-02-16 rows with form_fill 428.7/429.9/429.9/428.9 s).
- In standardized feature space, every row from a repeat applicant has its own resubmission as its nearest neighbour (100%). The largest within-person NN distance is 0.069. The smallest NN distance for a one-off applicant is 0.281.
- So a 1-NN classifier with a random split mostly finds the same person's other application in the training set and copies that label. That measures memorisation, not fraud detection. Dropping the identity columns as features was correct, but the same columns are also what you need to **group** the split.

## Evidence (each fix in isolation)

| Variant | Holdout, seed 1137 | 200 random splits, mean ± sd | LOO / LOGO (`cross_val_predict`) |
|---|---|---|---|
| Shipped pipeline (as is) | **0.9342** | 0.9453 ± 0.0121 | 0.9594 |
| Decoy 1 fixed: scaler inside a Pipeline | 0.9367 | 0.9449 ± 0.0122 | 0.9594 |
| Decoy 2 fixed: stratified split | 0.9418 | 0.9452 ± 0.0136 | 0.9594 (moot under LOO) |
| Decoys 1 + 2 fixed | 0.9392 | 0.9450 ± 0.0137 | 0.9594 |
| **Real bug fixed: split by person** | 0.7995 | 0.7677 ± 0.0336 | 0.7671 |
| **All three fixed** | 0.8021 | 0.7673 ± 0.0338 | **0.7666** |

- The decoy fixes move the score by at most 0.8 pp at seed 1137, well inside the ±1.2 pp split-to-split noise. Their 200-split means and LOO scores are unchanged.
- The grouped fix drops the score by about 18–19 pp. The majority-class baseline is 0.7088, so the real model adds only about 6 pp over always predicting "legit".

## Corrected pipeline

```python
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("applications.csv")
X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
y = df["label"]
# Identity fields are still not features, but they define who is who for the CV split.
person = df.groupby(["date_of_birth", "citizenship"]).ngroup()

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))  # scaler refit per fold
pred = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=person)
print(f"Accuracy: {(pred == y).mean():.4f}")   # 0.7666
```

## True figure
Leave-one-person-out with the shipped model gives 1511 of 1971 correct, **0.766616**. As a whole percent that is **77** when rounded (flag **`SIGMOID_77`**), or 76 if truncated. Keeping the scaler outside CV (the decoy) gives 0.7671, which leads to the same flag either way.
