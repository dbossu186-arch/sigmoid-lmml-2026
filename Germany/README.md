# GERMANY "Black Box": method

`main.py` uses at most 149 of the 150 queries (146 on the practice function). It assumes the task family
`f(x) = g(x) + J*[w.x >= t]`. Here `g` is smooth: an intercept, per-variable
main effects and pairwise products. The step is one linear half-space.
The code does not hard-code which variables do what. It finds every part of
the structure from data.

## Pipeline

1. **Design (80 queries).** The run queries all 32 cube corners and 48
   interior points from a scrambled Latin hypercube. A half-space that is not
   empty always contains a vertex of the cube, so the corners always include
   at least one point on each side of the step. The interior points remove an
   ambiguity that the corners alone cannot resolve. On {0,1}^5, a step on a
   sub-cube (for example `x1+x4>=1.7`) gives the same values as a bilinear
   term `J*x1*x4`. The interior points tell the two apart.
2. **Exact two-level regression (0 queries).** The data has no noise, so the
   right labelling of active and inactive points fits `y = A beta + J z`
   exactly. Here `A` is the smooth basis and `z` is the step indicator.
   - The code builds the annihilator `M = I - A A^+` once. For about 9,000
     candidate normals, it then computes the exact least-squares residual of
     every split "top-k points along d" in O(n^2) per direction, using
     cumulative sums. The candidates are axis-aligned normals, 2-variable fans,
     random 3-, 4- and 5-variable directions, and a local hill-climb.
   - The best splits are refined by alternating "fit g and J" with "relabel
     each point to the nearer of g and g+J".
   - The basis grows from linear + products (16 terms) to + squares to
     + cubes. It stops at the first level that fits to 5e-5.
   - Plain L1 or LS robust fits fail when 30-50% of the points sit in the step
     region, because the corners have high leverage. The scan does not have
     this problem.
3. **Boundary localisation (about 60 queries).**
   - With `g` and `J` known, each new query gives one bit: is `f - g` closer
     to 0 or to J?
   - Phase 1 bisects up to 7 nearby, spread-out pairs of points with opposite
     labels, down to a width of 0.008.
   - Phase 2 projects spread-out points onto the rough plane and steps along
     its normal (0.02, 0.05, 0.12) until the label flips. It then bisects down
     to a width of 0.0015.
   - A weighted total-least-squares plane is fitted through the crossing
     points. Components below 6% of the largest are set to zero and the plane
     is refitted. The variables that remain are the **gate variables**.
4. **Final readout.**
   - `g` and `J` are refitted jointly on all queries, with labels taken from
     the plane. Points within 0.004 of the plane are left out.
   - The step is reported as the side of the plane with less volume (Monte
     Carlo estimate). If needed, the code rewrites `g + J[A] = (g+J) - J[not A]`
     to match.
   - A variable is **relevant** if its main effect varies by more than 0.01
     over [0,1], if it is in the interaction, or if it is a gate variable.
   - Exactly one **interaction** is reported: the product with the largest
     |coefficient|. Its coefficient is on the monomial `xi*xj`, which equals
     the coefficient of the centred form.
   - `predictions` and `model_expression` come from the same fitted model.

## Files

- `main.py`: the submission. It uses stdlib and numpy only, and sets BLAS to
  one thread.
- `run_eval.py`: runs `data/evaluation_script.py` under `python -I`. Isolated
  mode drops the script directory from `sys.path`, which the evaluator needs
  for `from harness import ...`.
  Usage: `python -I run_eval.py main.py --eval-type train`.
- `stress_test.py`: generates random functions from the family and scores
  `main.py` on each one. It uses the evaluator's own `evaluate()` and the
  real `harness.run`, so the 150-query budget and the 50 s timeout apply.
  - Default ranges: 1-2 inert variables; interaction coefficient ±0.6..5,
    centred at 0.5 or off-centre; 1-3 gate variables (sometimes variables that
    otherwise look inert); active volume 2-48%; jump ±0.4..3. About 25% of the
    functions have non-linear main effects (quadratic, cubic, sin or exp).
  - `--hard`: 0-2 inert variables, effects down to 0.05, interaction ±0.2..6,
    jump ±0.1..4, 1-5 gate variables, and an active volume that is tiny
    (0.2-2%), normal, or a majority (52-95%).
  - Usage: `python -I stress_test.py -n 400 -j 16 [--hard] [--practice]`.

## Results

Practice (`--eval-type train`, Python 3.11 + numpy 2.2.4 on the Mac):
**100/100**. All four checks got full marks, and the run used 146 queries in
about 1 s.

Stress sweeps (run on a 16-core Linux machine: Python 3.14, numpy 2.2.4, 16 workers):

| sweep | n | min | median | mean | share >= 70 | max queries | max time |
|---|---|---|---|---|---|---|---|
| default family | 500 | 96 | 100 | 100.0 | 100% | 146 | 11.4 s* |
| `--hard` | 1000 | 89 | 100 | 97.9 | 100% | 149 | 11.0 s* |

\* These times are on a machine shared with other jobs and heavily loaded
(load average about 60). A single run takes 1 to 2.5 s on the Mac.

The raw results are in `stress_results_default.json` and
`stress_results_hard.json`.

Query budget: 80 design queries, about 40 for phase-1 bisection, and about
25 for phase-2 perpendicular probing. The run stays at or below 149 and keeps
one query in reserve. Each phase checks `oracle.remaining` before it spends.

## Known failure modes

- **Active region larger than half the cube** (`--hard`, -7 each time).
  `g + J[w.x>=t]` and `(g+J) - J[w.x<t]` give the same values at every
  point, so the data cannot say which side the true formula calls "active".
  The code reports the side with less volume. When the hidden formula puts
  the step on the larger side, the orientation check fails. Every other
  check, and the predictions, are unaffected.
- **Non-polynomial main effects** (sin/exp; outside the family the evaluator
  describes, which has only linear main effects). A cubic fit leaves about
  0.005-0.015 of residual. In rare cases this leaks a main effect above 0.01
  onto an inert variable (-4).
- A function with **no step at all**, or with more than one step, is outside
  the model. The code would still report its single best half-space.

## Compatibility note
`main.py` was validated on Python 3.11 + numpy 2.2.4 (the pinned version) and also on Python 3.14 + numpy 2.5.4
(a scalar conversion was made explicit so newer numpy does not raise). Practice score is 100/100 on both.
Re-check after the fix: 300 default-family functions min 100; 300 `--hard` functions min 89, median 100; 100% ≥ 70; max 149 queries.

Data files (harness, evaluator, challenge set) are the organisers' and are not included; put them in `data/` to run `run_eval.py` / `stress_test.py`.
