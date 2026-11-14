# Estonia — "This Data Is Perfect!" (Medium, 200 pts)

## Storyline
Estonia's Ministry of Digital Affairs fraud-detection team reports 93.42% accuracy at flagging fraudulent e-Residency applications. Pre-launch audit: three choices in the pipeline look questionable. Only one of them is why the number is fake.

## Objective
Find the three questionable choices in the pipeline and work out which one inflates the score. Test each one in isolation, and show that "fixing" the other two leaves the score unchanged (within run-to-run noise). Then fix the real bug and report the true performance.

## Flag
SIGMOID_<ACCURACY>: true accuracy of the corrected pipeline as a whole percent (0.8649 → SIGMOID_86). Measure:
- Keep the shipped model: StandardScaler + KNeighborsClassifier(n_neighbors=1) on the six numeric features.
- Leave-one-out CV on the full dataset: LeaveOneOut(), or its equivalent for your fix.
- One out-of-fold prediction per row with cross_val_predict; report fraction correct.

## Input
- applications.csv: submission_id, identity fields (date_of_birth, citizenship), six numeric features, label (1 fraud, 0 legit).
- project_pipeline.py: exact pipeline that produced 93.42%.
- Python 3.10–3.12, pandas==2.2.2, numpy==1.26.4, scikit-learn==1.5.1.
Data: https://drive.google.com/drive/folders/1fm_W5IUr4P4bnh2hyWDFbnYN-cZrhOkx?usp=sharing

## Expected output
Flag + link to short write-up (Markdown/notebook): the real bug and two decoys; evidence per decoy (score after "fixing" it); corrected pipeline; true figure.
