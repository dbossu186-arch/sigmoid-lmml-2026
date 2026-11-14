# Albania — "Looks like an anomaly!" (Medium, 200 pts)

## Storyline
Across Albania, thousands of concrete bunkers remain. A preservation team fitted one underground bunker with environmental and structural sensors to identify conditions that could threaten the site or the people inside.
The detector runs on a small monitoring device and receives one sensor reading at a time. It may remember what it has seen before, but cannot inspect future measurements.
The stream contains several kinds of abnormal behavior: sudden isolated spikes (point), slow drifts (gradual), and anomalies only suspicious when several moderate changes persist together (collective).

## Objective
Implement a streaming anomaly detector in detector.py:

```python
class Detector:
    def __init__(self):
        ...
    def update(self, reading: dict) -> float:
        ...
```
The harness calls update() once per timestep. State allowed, no access to future readings. Each call returns a finite score in [0.0, 1.0].

## Flag
SIGMOID_<WORD>, released automatically by the platform when final evaluation score >= 70.00/100.

## Input
CSV; `reading` dict has: timestamp, bunker_zone, temperature_c, humidity_pct, vibration_g, acoustic_db, air_pressure_kpa, co2_ppm.
Evaluator: Python 3.11, numpy==2.2.4, pandas==2.2.3, scikit-learn==1.6.1 + stdlib, CPU, 60-second timeout, no internet, no extra packages.

## Expected output
Submit detector.py (accepted .py, max 10 MB).

## Scoring
Scores thresholded at 0.50. Computes overall F1, recall on point anomalies, gradual drifts, collective anomalies.
score = 100 × (0.80·F1 + 0.05·point_recall + 0.075·drift_recall + 0.075·collective_recall), clamped 1–100.
Pass >= 70.00 → 200 points.

## Data
Google Drive folder: https://drive.google.com/drive/folders/14AMHetQATgeB-wSm0-Da10NffyxOGQVo?usp=sharing
