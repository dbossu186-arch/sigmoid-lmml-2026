# GERMANY: Black Box (Hard)

Source: https://sigmoid.academy/tasks (popup text copied 2026-11-14)

## Storyline
A sensor rig came back from the field with its documentation lost. What survived is the rig itself: five dials, each turnable from 0 to 1, and one readout. Turn the dials, read the number. That is the entire interface.

The lab that built it is gone, the schematic with it. The rig now sits sealed in a test lab in Stuttgart, and your team never touches it. You hand in a procedure (a Python script), and the lab's technician runs it exactly once against the rig. The readout is expensive to take: your procedure gets 150 readings, the rig counts them itself, and a procedure that asks for a 151st is thrown out.

Management does not want a curve that happens to fit. They want the wiring diagram: which dials are actually connected to anything, which ones are decorative, how the connected ones influence each other, and where the device stops behaving smoothly.

> **WARNING.** One team already tried this. Their procedure turned each dial on its own, held the others at the middle, and got five beautiful straight lines with residuals near 1e-11. It wrote up an additive linear model, marked it "high confidence", and was wrong about most of the device. Their report is not a cautionary tale about sloppy work; their arithmetic was flawless. The rig was built so that careful one-dial-at-a-time work produces a confident wrong answer.

## Objective
A hidden, deterministic, noise-free function f(x1..x5) on [0,1]^5 is wired into the grader. Write a `main.py` whose `solve(oracle, points)` uses **at most 150 queries** to recover the function's structure, not just its numbers:
- which inputs matter and which are inert,
- how the relevant inputs interact (the function is not purely additive),
- where its behaviour changes: there is at least one discontinuity, a surface in the input space across which the output jumps.

It must also predict the function on 300 held-out points. Prediction is worth only 30 of 100 points, on purpose.

The script runs once, unattended, against the hidden function. You never see a reading from it before that run, so main.py must discover the structure on its own.

## Flag
`SIGMOID_<WORD>`. Organisers run: `python3 evaluation_script.py main.py --eval-type test`. A score ≥ 70 prints `FLAG RELEASE: SIGMOID_...`. The practice run (`--eval-type train`) never prints a flag.

## Task input
| File | What it is |
|---|---|
| harness.py | the Oracle your solve() receives, and the runner the graders use |
| practice_function.py | a practice black box from the same family (main effects, one interaction, one hard step) wired differently. Test against it as often as you like |
| evaluation_script.py | the grader. `--eval-type train` scores main.py against the practice function |
| challenge/challenge_set.json | the 300 points to predict; solve() receives the same list as `points` |
| submission.example.json | shape of the dict solve() returns, with placeholder values |
| baseline/naive_owat.py | a complete valid main.py that walks into the trap (~20/100). Start from its shape, not its strategy |

The graded function is not given.

Example:
```python
def solve(oracle, points):
    y = oracle.ask([0.5, 0.5, 0.5, 0.5, 0.5])                    # 1 query
    ys = oracle.ask_many([[1, 1, .5, .5, .5],
                          [0, 0, .5, .5, .5]], phase="corners")   # 2 queries
    oracle.spent, oracle.remaining                                # bookkeeping
    ...
    return {...}                       # see submission.example.json
```
Practice: `python3 evaluation_script.py main.py --eval-type train` (report shows score, per-check breakdown and the practice function itself).

## Rules enforced by the harness
- Inputs x1..x5 each in [0,1]. A point outside the domain raises ValueError and costs nothing.
- Every point that reaches the harness costs exactly one query. The Oracle caches answers; re-asking a known point is free.
- Deterministic, noise-free. Output rounded to 6 decimals.
- A 151st query ends the run: script stopped, score 0. The exception cannot be caught and recovered from. Watch `oracle.remaining`.
- Run is stopped after **50 seconds**. Crashing, timing out, or returning a non-dict scores 0.
- main.py is one file: no helper modules travel with it. No network.
- print() is fine (goes to stderr). Queries are counted by the harness.
- Python 3.10+. Harness, grader and baseline use only the standard library. main.py may additionally import **numpy==2.2.4** (requirements.txt); nothing else outside the standard library.

## Expected output: dict returned by solve()
| Field | Content |
|---|---|
| relevant_variables / irrelevant_variables | classify all five inputs; a variable left out of both scores zero |
| interactions | list of {variables, type, coefficient}; coefficient = multiplier on the xi*xj monomial after expanding the expression |
| discontinuities | list of {gate_variables, boundary, jump}; boundary structured: {form, coefficients: {x1..x5}, threshold, active_when} |
| model_expression | one Python-arithmetic string in x1..x5, e.g. `"1.0 + 0.5*x1 + ... + (2.0 if ... else 0.0)"` |
| predictions | 300 floats, one per point, in order |
| notes | how the budget was spent and why you believe this structure |

The boundary is scored by orientation and offset; any positive rescaling of coefficients and threshold together is accepted (x2 + x4 >= 1.6 ≡ 10*x2 + 10*x4 >= 16).

## Scoring (out of 100, on the hidden function)
| Check | Points | How |
|---|---|---|
| budget | gate | > 150 queries ⇒ run stopped, scores 0 |
| variables | 20 | 4 per input correctly called relevant or inert |
| interaction | 20 | interacting pair (12) + coefficient within 15% (8), minus 3 per spurious interaction, capped at −6 |
| discontinuity | 30 | detected (8) + gate variables (10) + boundary orientation and offset (7) + jump size within 15% (5) |
| prediction | 30 | RMSE on 300 held-out points (20) + RMSE restricted to the discontinuous region (10) |

RMSE scoring is log-scaled: ≤ 0.01 full marks, ≥ 2.0 none, interpolated between. If predictions is missing, model_expression is evaluated on the challenge set. Challenge points within 0.005 of the true switching surface are dropped from scored RMSE. 70 of 100 points are structural.

- Score ≥ 70: 300 points (PASS), flag released
- Score < 70: 0 points (FAIL)

## Submission
Upload, accepted type .py, max 10 MB. Hints panel showed "Error while loading hints" (do not unlock without asking).

## Data
Google Drive folder ID `1XRPCzOo2Jz1ymh80bvbmTG8OXH-EI_A_` → `data/`
