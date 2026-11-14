"""Pick, for each question independently, the cheapest (tier, style, fewshot, context)
choice that still covers every required fact. Questions are independent and each has
only 2*2*2*2 = 16 options, so exhaustive search gives the exact optimum.

    python solve.py   # writes routing_config.json and prints the per-question table
"""
import itertools
from pathlib import Path

import model_utils as mu

fb = mu.load_fact_bank()
config, rows = {}, []
for qid, q in fb.items():
    best = None
    for tier, style, fewshot, context in itertools.product(mu.TIERS, mu.STYLES, (False, True), (False, True)):
        call = mu.build_call(q, tier, style, fewshot, context)
        if mu.question_passes(q["slots"], call["answer"]) and (best is None or call["tokens"] < best[0]):
            best = (call["tokens"], dict(tier=tier, style=style, fewshot=fewshot, context=context))
    config[qid] = best[1]
    naive = mu.build_call(q, "large", "verbose", True, False)["tokens"]
    rows.append((qid, naive, best[0], best[1]))

mu.save_json(config, Path("routing_config.json"))
res = mu.run_config(config, fb)
naive_total = mu.run_config(mu.load_json(mu.EXAMPLE_NAIVE_CONFIG_PATH), fb)["total_tokens"]
for qid, n, t, c in rows:
    print(f"{qid:8s} naive {n:4d} -> {t:3d}  {c['tier']:5s} {c['style']:7s} fewshot={c['fewshot']!s:5s} context={c['context']}")
print(f"total: naive {naive_total} -> {res['total_tokens']} (budget {mu.TOKEN_BUDGET}); "
      f"quality={res['quality_gate']} budget={res['budget_gate']}")
