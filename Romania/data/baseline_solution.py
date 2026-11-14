"""The workflow you start from.

1. load the training records and the sealed records;
2. fit two usual single models (logistic regression, SVM), one at a time, and look at
   their cross-validation accuracy;
3. predict the sealed records with the best of them;
4. write predictions.csv and ask the evaluator.

    python baseline_solution.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
CHANNELS = [f"c{j + 1:02d}" for j in range(48)]

# 1. Load ---------------------------------------------------------------------
train = pd.read_csv(HERE / "dataset" / "train.csv")
sealed = pd.read_csv(HERE / "dataset" / "sealed.csv")
print("train:", train.shape, " sealed:", sealed.shape)
print(train.head(3).to_string())

X, y = train[CHANNELS].to_numpy(), train.vrancea.to_numpy()
X_sealed = sealed[CHANNELS].to_numpy()

# 2. Two single models, one at a time ---------------------------------------------
models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    "SVM (rbf)": make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale")),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = {}
print("\n5-fold cross-validation accuracy on the training records:")
for name, model in models.items():
    scores[name] = cross_val_score(model, X, y, cv=cv).mean()
    print(f"  {name:20s} {scores[name]:.3f}")

# 3. Predict the sealed records with the best single model --------------------
best = max(scores, key=scores.get)
print(f"\nbest single model: {best}")
pred = models[best].fit(X, y).predict(X_sealed)
out = pd.DataFrame({"record_id": sealed.record_id, "vrancea": pred.astype(int)})
out.to_csv(HERE / "predictions.csv", index=False)
print("wrote predictions.csv:", out.vrancea.value_counts().to_dict())

# 4. Ask the evaluator --------------------------------------------------------
print("\nevaluator says:")
sys.stdout.flush()
subprocess.run([sys.executable, str(HERE / "evaluation_script.py"), str(HERE / "predictions.csv")], check=False)
