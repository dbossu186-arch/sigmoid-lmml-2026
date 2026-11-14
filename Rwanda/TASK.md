# Rwanda — "The Drying Floor at Karongi" (Easy, 100 pts)
Karongi coffee cooperative automated its logbook as a LangGraph state machine: arrive → measure → dry (one pass) → grade. The measure↔dry loop should end the moment a lot is ready (moisture within spec). It never ends: lots already in spec keep drying, lots needing drying never get marked ready; every batch bounces between the same two stations forever. The drying itself isn't the problem — the inspection step never tells the loop "this one's done".

Objective: run graph.py against dataset/batches.json; the graph must converge correctly and reach END.
Flag: SIGMOID_<WORD> (printed by the runner on a full pass).
Input: dataset/batches.json, 8 lots, e.g. {"batch_id": "nyamasheke", "moisture": 12.0}. No hidden split — evaluation uses exactly this file. Accepted range MOISTURE_MIN 10.5, MOISTURE_MAX 12.5; some lots need zero passes, others several. No special-casing batch_id.
Env: Python 3.10–3.12, langgraph==1.2.16, langchain-core==1.6.12. No API keys/GPU/internet; no model calls.
graph_reference.py is alongside — don't edit or submit it; evaluation_script.py uses it to confirm your fix only touches what's needed and rejects a graph.py that changes anything else.
Expected output: (1) corrected graph.py, (2) run log, e.g. `python evaluation_script.py | tee run_log.txt`. Submit flag + URL (GitHub/Drive) containing both files.
Scoring: all-or-nothing. PASS if every batch finishes within recursion_limit=18 and its final moisture, drying-pass count and grade match the correct outcome. FAIL if any batch exceeds the budget, finishes without approved, mismatches, or graph.py changes more than needed.
Data: https://drive.google.com/drive/folders/1c1tXsOOX7kX3H881Qv8l6VXv7t6xRnp8
