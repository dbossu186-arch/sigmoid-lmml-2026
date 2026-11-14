# Rwanda — The Drying Floor at Karongi

## Bug
`inspect_batch` works out `ready = MOISTURE_MIN <= moisture <= MOISTURE_MAX` (inclusive on both ends, which is correct), then throws the result away. It only returns `{"inspections": ...}`, so `approved` stays `False`. `route_after_inspection` therefore always sends the lot to `dry_batch`, and the measure/dry loop runs until it hits the recursion limit.

## Fix (one line, inside `inspect_batch` only)
```diff
-    return {"inspections": state["inspections"] + 1}
+    return {"inspections": state["inspections"] + 1, "approved": ready}
```
Everything else matches `graph_reference.py`. The original file is kept at `data/graph.py`.

## Run
`evaluation_script.py`, `graph_reference.py` and `dataset/` are copied here from `data/` because the evaluator imports `graph.py` from its own folder.
```
uv venv -p 3.12 .venv && uv pip install langgraph==1.2.16 langchain-core==1.6.12
.venv/bin/python evaluation_script.py | tee run_log.txt
```
All 8 batches pass (see `run_log.txt`).

## Flag
The flag is printed by the evaluator on a full pass (see run_log.txt).
