# Senegal — "The Teranga FM Token Diet" (Medium, General AI) — Sigmoid LMML 2026

## Storyline (summary)
Teranga FM, a community radio station in Kaolack in Senegal's Peanut Basin, runs a text helpline for farmers and listeners. Answers go out over a billed SMS gateway, and the LLM provider bills per token for both prompt and completion. The intern sends every question to the biggest model, verbose, with a few-shot block attached. That is accurate but blows the monthly token allowance. The job: answer the same battery of questions just as accurately for far fewer tokens.

## Objective
There are 20 fixed questions (dataset/fact_bank.json) and an offline, deterministic simulator of the LLM gateway (model_utils.py). For each question choose:
- tier: cheap or large
- style: verbose or terse
- fewshot: attach the fixed few-shot block or not
- context: attach the question's context snippet (if it has one) or not

The costs and outputs are fully determined by model_utils.py and fact_bank.json; read those, not this document.

- Quality gate: every one of the 20 answers must include every required fact. No partial credit.
- Budget gate: total tokens (prompt + completion) across all 20 answers must be ≤ 1200.
- The naive config (large / verbose / fewshot / no context) passes quality at 3626 tokens.

## Flag
Format `SIGMOID_<WORD>`. It is released by the organizer when both gates pass.

## Inputs
| File | Purpose |
|---|---|
| fact_bank.json | 20 questions, their required facts, and context snippets |
| example_naive_config.json | the intern's config |
| model_utils.py | gateway simulator (templates, token counter, quality checker); the grader uses this exact file |
| main_template.py | template for main.py: reads routing_config.json, writes answers.json |
| baseline_solution.ipynb | shows how to produce and self-check answers locally |

The scripts expect dataset/fact_bank.json and dataset/example_naive_config.json next to them.

## Environment
Python 3.10–3.12, standard library only. No GPU, no API keys, no internet. Each run takes well under a second.

## Expected output
- main.py: a copy of main_template.py with the I/O contract unchanged.
- routing_config.json: one entry per question id, each {"tier", "style", "fewshot", "context"}.
- Any script used to choose the routes.

Package all of it behind a GitHub or Drive URL. The platform form has a Link field and a .json upload (max 10 MB). Grading rebuilds every answer from routing_config.json using the organizer's model_utils.py and fact_bank.json.

## Scoring
Both gates pass: 200 points (PASS) and the flag is released. Either fails: 0 points.

Data: https://drive.google.com/drive/folders/1QuDKEu05swvKZiz_SDCzyLaEyKNT8iud (folder ID `1QuDKEu05swvKZiz_SDCzyLaEyKNT8iud`)
Hints: 3 locked (penalty), none unlocked.
