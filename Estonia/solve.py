"""
Sigmoid -- Estonia "This Data Is Perfect!"  Reproduces every number in README.md.

Run:  python solve.py          (pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.1)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GroupShuffleSplit, LeaveOneGroupOut, LeaveOneOut,
                                     cross_val_predict, train_test_split)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data" / "applications.csv"
df = pd.read_csv(DATA)
FEATS = ["form_fill_seconds", "doc_scan_quality", "photo_match_score",
         "email_domain_age_days", "ip_geo_distance_km", "address_history_years"]
X = df[FEATS].to_numpy()
y = df["label"].to_numpy()
# A person = (date_of_birth, citizenship). Used ONLY to define CV groups, never as a feature.
person = df.groupby(["date_of_birth", "citizenship"]).ngroup().to_numpy()


def knn():
    return KNeighborsClassifier(n_neighbors=1)


def pipe():
    return make_pipeline(StandardScaler(), knn())


def holdout(scale_outside=True, stratify=False, grouped=False, seed=1137):
    """Shipped holdout protocol with each questionable choice toggleable."""
    Xin = StandardScaler().fit_transform(X) if scale_outside else X
    model = knn() if scale_outside else pipe()
    if grouped:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(Xin, y, person))
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed,
                                  stratify=y if stratify else None)
    model.fit(Xin[tr], y[tr])
    return accuracy_score(y[te], model.predict(Xin[te]))


def loo(scale_outside=True, grouped=False):
    """Flag protocol: one out-of-fold prediction per row via cross_val_predict."""
    Xin = StandardScaler().fit_transform(X) if scale_outside else X
    model = knn() if scale_outside else pipe()
    if grouped:
        pred = cross_val_predict(model, Xin, y, cv=LeaveOneGroupOut(), groups=person)
    else:
        pred = cross_val_predict(model, Xin, y, cv=LeaveOneOut())
    return (pred == y).mean(), int((pred == y).sum())


def spread(**kw):
    accs = [holdout(seed=s, **kw) for s in range(200)]
    return np.mean(accs), np.std(accs)


if __name__ == "__main__":
    print(f"rows={len(df)}  fraud rate={y.mean():.4f}  majority baseline={1 - y.mean():.4f}")
    sizes = pd.Series(person).value_counts()
    print(f"distinct persons={person.max() + 1}  submissions/person: "
          f"{sizes.value_counts().sort_index().to_dict()}")
    print("label constant within every person:",
          bool((df.assign(p=person).groupby('p').label.nunique() == 1).all()))
    Xs = StandardScaler().fit_transform(X)
    d, i = NearestNeighbors(n_neighbors=2).fit(Xs).kneighbors(Xs)
    multi = pd.Series(person).map(sizes).to_numpy() > 1
    print(f"rows whose nearest neighbour is the same person: {(person[i[:, 1]] == person).mean():.4f} "
          f"(of rows with resubmissions: {(person[i[:, 1]] == person)[multi].mean():.4f})")
    print(f"max NN distance within person {d[multi, 1].max():.3f}  vs  min NN distance "
          f"for one-off applicants {d[~multi, 1].min():.3f}\n")

    rows = [
        ("Shipped pipeline (as is)",                 dict()),
        ("Decoy 1 fixed: scaler fit inside training", dict(scale_outside=False)),
        ("Decoy 2 fixed: stratified split (imbalance)", dict(stratify=True)),
        ("Decoys 1+2 fixed",                          dict(scale_outside=False, stratify=True)),
        ("REAL BUG fixed: split by person",           dict(grouped=True)),
        ("All three fixed",                           dict(scale_outside=False, grouped=True)),
    ]
    print(f"{'variant':46s} {'holdout@1137':>12s} {'200 splits mean+-sd':>20s} {'LOO / LOGO':>11s}")
    for name, kw in rows:
        h = holdout(**kw)
        m, s = spread(**kw)
        lkw = {k: v for k, v in kw.items() if k != "stratify"}  # stratification is moot under LOO
        a, n = loo(**lkw)
        print(f"{name:46s} {h:12.4f} {m:13.4f} +- {s:.4f} {a:11.4f}")

    acc, n = loo(scale_outside=False, grouped=True)
    print(f"\nTRUE ACCURACY (Pipeline(StandardScaler, 1-NN), LeaveOneGroupOut by person, "
          f"cross_val_predict): {n}/{len(y)} = {acc:.6f}")
    print(f"whole percent: truncated -> SIGMOID_{int(acc * 100)}, rounded -> SIGMOID_{round(acc * 100)}")
