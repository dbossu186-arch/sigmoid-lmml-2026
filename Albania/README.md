# Albania: Looks like an anomaly! (Sigmoid LMML 2026, Medium)

A streaming anomaly detector for the bunker sensor stream. It works online, uses no future data, needs only the Python standard library, and costs O(1) per reading.

- **What "normal" means (`detector.py`):** each sensor is a slowly varying baseline plus white noise.
  - The baseline is tracked per feature with a Holt filter (level plus bounded trend), so slow daily cycles are followed.
  - While an anomaly is flagged, the filter is frozen and only extrapolated, so a drift isn't learned as normal.
  - The noise scale per feature is an EWMA of the squared standardised residuals.
- **Detectors**, all on the standardised residuals:
  - **point:** the largest |z| in the current reading.
  - **drift:** a signed EWMA of z per feature, confirmed by a faster EWMA so the alarm ends promptly.
  - **collective:** an EWMA of the multivariate energy, which catches several moderate deviations at once.
- **`harness.py`:** replays a stream through the detector and computes the task score locally.
- **`tune/`:** a synthetic-stream generator plus a parameter search.
