"""Local harness: replay a stream through Detector and compute the task score.

usage: python harness.py [--start N] [--end M] [--repeat K] [--strings]
"""
import argparse
import csv
import importlib.util
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_detector(path=os.path.join(HERE, "detector.py")):
    spec = importlib.util.spec_from_file_location("detector_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(inp=os.path.join(HERE, "data", "train-input.csv"),
              out=os.path.join(HERE, "data", "train-output.csv")):
    with open(inp) as f:
        rows = list(csv.DictReader(f))
    with open(out) as f:
        labs = list(csv.DictReader(f))
    y = [int(r["is_anomaly"]) for r in labs]
    t = [r["anomaly_type"] for r in labs]
    return rows, y, t


def score(pred, y, types):
    tp = sum(1 for p, a in zip(pred, y) if p and a)
    fp = sum(1 for p, a in zip(pred, y) if p and not a)
    fn = sum(1 for p, a in zip(pred, y) if not p and a)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    def trec(name):
        idx = [i for i, k in enumerate(types) if k == name]
        return (sum(pred[i] for i in idx) / len(idx)) if idx else 0.0

    pr, dr, cr = trec("point"), trec("drift"), trec("collective")
    s = 100 * (0.80 * f1 + 0.05 * pr + 0.075 * dr + 0.075 * cr)
    s = min(100.0, max(1.0, s))
    return dict(score=s, f1=f1, prec=prec, rec=rec, point=pr, drift=dr,
                collective=cr, tp=tp, fp=fp, fn=fn)


def run(mod, rows, params=None, as_strings=False):
    det = mod.Detector(params) if params is not None else mod.Detector()
    out = []
    for r in rows:
        rr = dict(r) if as_strings else {
            k: (v if k in ("timestamp", "bunker_zone") else float(v))
            for k, v in r.items()}
        s = det.update(rr)
        assert isinstance(s, float) and math.isfinite(s) and 0 <= s <= 1, s
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--strings", action="store_true")
    ap.add_argument("--det", default=os.path.join(HERE, "detector.py"))
    a = ap.parse_args()
    mod = load_detector(a.det)
    rows, y, t = load_data()
    rows, y, t = rows[a.start:a.end], y[a.start:a.end], t[a.start:a.end]
    t0 = time.time()
    sc = run(mod, rows * a.repeat, as_strings=a.strings)
    dt = time.time() - t0
    pred = [s >= 0.5 for s in sc[:len(rows)]]
    res = score(pred, y, t)
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in res.items()))
    print(f"rows={len(rows) * a.repeat} runtime={dt:.3f}s")


if __name__ == "__main__":
    main()
