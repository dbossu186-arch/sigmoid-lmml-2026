# Senegal: The Teranga FM Token Diet

Routing config for the Teranga FM helpline simulator: **all 20 quality bars pass at 1160 tokens**, under the 1200 cap. The intern's naive config scores 3626.

## Files
- `routing_config.json`: the submission, one `{tier, style, fewshot, context}` entry per question.
- `main.py`: unchanged copy of `main_template.py`. It reads `routing_config.json` and writes `answers.json`.
- `solve.py`: the script that produced the config.
- `answers.json`: output of `python main.py`.

`model_utils.py` and the `dataset/` folder (`fact_bank.json`, `example_naive_config.json`) come from the organizer's materials and aren't redistributed here. Copy them next to `main.py` to reproduce:

```
python solve.py   # -> routing_config.json
python main.py    # quality: 20/20 PASS, tokens: 1160 / 1200 PASS
```

The output is the same on Python 3.11 and 3.14; it uses only the standard library.

## How the config was found
In `model_utils.py`, a question's cost and pass/fail depend only on that question's own choice, and the total is the plain sum over questions. The problem therefore splits into 20 independent problems with 2×2×2×2 = 16 options each. `solve.py` tries all 320 options and keeps the cheapest passing one for each question, which is the exact optimum. No cheaper passing config exists.

What the simulator rewards:

| Lever | Effect |
|---|---|
| `style: terse` | Always right. It drops the long instruction from the prompt and the INTRO/OUTRO boilerplate from the answer, and coverage doesn't change. |
| `fewshot: false` | Always right. The few-shot block costs ~30 tokens and never affects the answer. |
| `tier: cheap` | Much shorter preamble, but the cheap tier drops every fact slot marked `large_only`. |
| `context: true` | On the cheap tier it restores the `large_only` slots at the cost of the snippet's tokens. On the large tier it is pure waste. |

So for each question:
1. **No `large_only` slots:** cheap / terse / no context (questions 1, 3, 9, 11, 14, 15, 16, 19).
2. **`large_only` slots and a snippet shorter than the large-tier preamble:** cheap / terse / context (questions 2, 10, 13, 17, 18, 20).
3. **`large_only` slots and no usable snippet:** large / terse (questions 4–8 and 12).

## Results

| Config | Quality | Tokens |
|---|---|---|
| Naive: large / verbose / fewshot | 20/20 | 3626 |
| Everything terse, no fewshot, large | 20/20 | 1506 (over budget) |
| Everything cheap / terse | fails the questions that have `large_only` slots | — |
| **Per-question optimum (this repo)** | **20/20** | **1160** |
