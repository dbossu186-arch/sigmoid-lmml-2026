"""
Stress test for main.py.

  python -I stress_test.py --practice            # evaluator on practice fn
  python -I stress_test.py -n 500 -j 16          # random family sweep

Random hidden functions from the task family
    f = b0 + sum_i h_i(x_i) + c*(xi-a)*(xj-b) + J*[w.x >= t]
with varied inert sets, interacting pair/sign/size/centre, boundary
orientation / gate variables (1-3, possibly otherwise-inert) / offset
(active volume 2%..48%), jump sign/size, main-effect shapes (linear by
default, a share with quadratic/cubic/sin/exp shapes).

Each function is scored by the evaluator's own code (evaluation_script.
evaluate) on the real challenge points, running main.py through the real
harness.run (budget 150 enforced, 50 s timeout, separate process).
"""
import argparse
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, DATA)

import evaluation_script as ev   # noqa: E402
import harness                    # noqa: E402

VARS = ["x1", "x2", "x3", "x4", "x5"]
SHAPES = ["linear", "quad", "cubic", "sin", "exp"]


def make_function(seed, nonlinear_share=0.25, hard=False):
    """hard=True widens every range: weak effects, tiny/majority regions,
    small jumps, 1-5 gate variables, off-centre interactions."""
    R = random.Random(seed)
    b0 = R.uniform(-2, 3)
    idx = list(range(5))
    # interaction pair
    pi, pj = sorted(R.sample(idx, 2))
    c = R.choice([-1, 1]) * (R.uniform(0.2, 6.0) if hard else R.uniform(0.6, 5.0))
    ca = 0.5 if R.random() < 0.6 else R.uniform(0.2, 0.8)
    cb = 0.5 if R.random() < 0.6 else R.uniform(0.2, 0.8)
    # gate variables (1..3), may include otherwise-inert ones
    ng = (R.choices([1, 2, 3, 4, 5], weights=[2, 4, 3, 2, 1])[0] if hard
          else R.choices([1, 2, 3], weights=[2, 5, 2])[0])
    gate = sorted(R.sample(idx, ng))
    w = [0.0] * 5
    for g in gate:
        w[g] = R.choice([-1, 1]) * R.uniform(0.5, 2.0)
        if R.random() < 0.5:
            w[g] = math.copysign(1.0, w[g])
    # main effects: inert vars are those not in interaction/gate with prob
    mains = [None] * 5
    n_inert_target = R.choice([0, 1, 1, 2] if hard else [1, 1, 2])
    free = [i for i in idx if i not in (pi, pj) and i not in gate]
    R.shuffle(free)
    inert = set(free[:n_inert_target])
    for i in idx:
        if i in inert:
            continue
        if i in gate and i not in (pi, pj) and R.random() < 0.4:
            continue                      # gate-only variable (like practice x4)
        if i in (pi, pj) and R.random() < 0.2:
            continue                      # interaction-only variable
        amp = R.choice([-1, 1]) * (R.uniform(0.05, 2.5) if hard else R.uniform(0.15, 2.0))
        shape = "linear"
        if R.random() < nonlinear_share:
            shape = R.choice(SHAPES[1:])
        mains[i] = (shape, amp)
    # threshold for an active volume fraction
    frac = R.uniform(0.02, 0.48)
    if hard:
        frac = R.choice([R.uniform(0.002, 0.02), R.uniform(0.02, 0.48),
                         R.uniform(0.52, 0.95)])
    MC = random.Random(seed + 1)
    s = sorted(sum(w[k] * MC.random() for k in range(5)) for _ in range(20000))
    t = s[int((1 - frac) * len(s))]
    J = R.choice([-1, 1]) * (R.uniform(0.1, 4.0) if hard else R.uniform(0.4, 3.0))

    def h(i, x):
        sh, a = mains[i]
        if sh == "linear":
            return a * x
        if sh == "quad":
            return a * (x - 0.3) ** 2
        if sh == "cubic":
            return a * (x ** 3 - x)
        if sh == "sin":
            return a * math.sin(2.0 * x)
        return a * math.exp(x) / 2

    def f(x):
        y = b0
        for i in idx:
            if mains[i] is not None:
                y += h(i, x[i])
        y += c * (x[pi] - ca) * (x[pj] - cb)
        if sum(w[k] * x[k] for k in idx) >= t:
            y += J
        return y

    relevant = sorted({i for i in idx if mains[i] is not None} | {pi, pj} | set(gate))
    M = types.SimpleNamespace()
    M.VAR_NAMES = VARS
    M.TRUTH = {
        "linear": {VARS[i]: (mains[i][1] if mains[i] else 0.0) for i in idx},
        "shapes": {VARS[i]: (mains[i][0] if mains[i] else None) for i in idx},
        "interaction": {"variables": (VARS[pi], VARS[pj]), "form": "centered_product",
                        "coefficient": c, "center": (ca, cb)},
        "discontinuity": {"gate_variables": tuple(VARS[g] for g in gate),
                          "boundary": {VARS[k]: w[k] for k in idx},
                          "threshold": t, "jump": J, "active_frac": frac},
        "relevant": [VARS[i] for i in relevant],
        "irrelevant": [VARS[i] for i in idx if i not in relevant],
    }
    M.gate_active = lambda x: sum(w[k] * x[k] for k in idx) >= t
    M.f = f
    M.evaluate = lambda x: round(f(x), 6)
    M.expression = lambda: f"seed {seed}"
    return M


def run_one(args):
    seed, script, nonlinear_share, hard = args
    HF = make_function(seed, nonlinear_share, hard)
    with open(os.path.join(DATA, "challenge", "challenge_set.json")) as fh:
        points = json.load(fh)["points"]
    ev.HF = HF
    t0 = time.time()
    out = harness.run(script, HF.evaluate, points, harness.BUDGET, harness.TIMEOUT)
    dt = time.time() - t0
    r = ev.evaluate(out, f"seed{seed}", points)
    tr = HF.TRUTH
    return {"seed": seed, "score": r["score"], "queries": r["queries_used"],
            "time": round(dt, 2), "failed": r["failed"], "flags": r["flags"],
            "breakdown": {k: v["points"] for k, v in r["breakdown"].items()},
            "detail": {k: v.get("detail") for k, v in r["breakdown"].items()},
            "nonlinear": any(s not in (None, "linear") for s in tr["shapes"].values()),
            "n_gate": len(tr["discontinuity"]["gate_variables"]),
            "active_frac": tr["discontinuity"]["active_frac"],
            "jump": tr["discontinuity"]["jump"],
            "truth": {"interaction": [tr["interaction"]["variables"],
                                      tr["interaction"]["coefficient"]],
                      "disc": tr["discontinuity"], "relevant": tr["relevant"],
                      "shapes": tr["shapes"]}}


def summarise(res, label):
    sc = sorted(r["score"] for r in res)
    if not sc:
        return
    n = len(sc)
    print(f"{label}: n={n} min={sc[0]:.1f} p10={sc[int(0.1*n)]:.1f} "
          f"median={statistics.median(sc):.1f} mean={statistics.mean(sc):.1f} "
          f">=70: {sum(s >= 70 for s in sc)/n:.1%}  max_q="
          f"{max(r['queries'] for r in res)} max_t={max(r['time'] for r in res):.1f}s")
    for k in ("variables", "interaction", "discontinuity", "prediction"):
        vals = [r["breakdown"].get(k, 0.0) for r in res]
        print(f"   {k:14s} mean {statistics.mean(vals):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("-j", type=int, default=os.cpu_count())
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--script", default=os.path.join(HERE, "main.py"))
    ap.add_argument("--nonlinear-share", type=float, default=0.25)
    ap.add_argument("--hard", action="store_true",
                    help="wider/harsher parameter ranges (see make_function)")
    ap.add_argument("--practice", action="store_true",
                    help="also run evaluation_script.py --eval-type train")
    ap.add_argument("--out", default=os.path.join(HERE, "stress_results.json"))
    a = ap.parse_args()

    if a.practice:
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_eval.py"),
                        a.script, "--eval-type", "train"])
    if a.n <= 0:
        return
    jobs = [(a.seed0 + k, a.script, a.nonlinear_share, a.hard) for k in range(a.n)]
    with ProcessPoolExecutor(a.j) as ex:
        res = list(ex.map(run_one, jobs))
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    summarise(res, "ALL")
    summarise([r for r in res if not r["nonlinear"]], "linear mains")
    summarise([r for r in res if r["nonlinear"]], "nonlinear mains")
    summarise([r for r in res if r["active_frac"] < 0.02], "active vol <2%")
    summarise([r for r in res if r["active_frac"] > 0.5], "active vol >50%")
    summarise([r for r in res if abs(r["jump"]) < 0.4], "|jump| < 0.4")
    for g in (1, 2, 3, 4, 5):
        summarise([r for r in res if r["n_gate"] == g], f"gate vars={g}")
    bad = sorted(res, key=lambda r: r["score"])[:8]
    print("worst:")
    for r in bad:
        print(f"  seed {r['seed']} score {r['score']} {r['breakdown']} "
              f"q={r['queries']} flags={r['flags']}")


if __name__ == "__main__":
    main()
