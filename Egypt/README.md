# Egypt — Always Valid, No Excuses

Converts OCR'd record cards (`key: value` segments separated by `|`, with scan noise)
into one strict JSON object per line.

Upload: `test-output.txt` (produced by `main.py` from `test-input.txt`).

## How it works (`main.py`)

1. Deterministic parser (`parse_record`) implementing the spec:
   - split on `|`; a segment is a field only if it contains `:` and the text before the
     **first** `:` (trimmed) is a simple word `[A-Za-z_][A-Za-z0-9_]*`; everything else
     (page 22, p. 128, 22, Chapter IX, - 45 -, [torn], (margin note), Fig 3, Plate II,
     ###, ~~~, * * *, see above, scanned 1993, end of page) is dropped;
   - key and value trimmed (also strips the CRLF in the test file);
   - `""` → null, exactly `true`/`false` → booleans, `-?(0|[1-9][0-9]*)` → integer
     (so `+5`, `007`, `-0`, `1.5`, `1e3`, `True`, `null` would stay strings — none occur
     in the test input), anything else → string (JSON-escaped by `json.dumps`).
   The parser's output is written to `test-output.txt` first, so a valid file exists even
   if the model step fails or is slow.
2. The oracle: Qwen3-1.7B Q8_0 (llama-cpp-python, `n_gpu_layers=-1`, greedy, few-shot,
   no-think, prompt-lookup speculative decoding) converts every record itself.
3. Each model line must parse as strict JSON and decode to exactly the parser's data
   (bool vs int distinguished). Agreeing lines are kept; anything else is repaired to the
   parser's result. The file is then rewritten. The model phase has a wall-clock budget
   (`EGYPT_MODEL_BUDGET`, default 45 s) so the run stays inside the 60 s limit; records the
   model did not reach use the parser result.

Env: `EGYPT_NO_MODEL=1` (rules only), `EGYPT_LOG=path.jsonl` (per-line model/rules log).

## Setup

```
python3.11 -m venv .venv            # Python 3.10-3.12
CMAKE_ARGS="-DGGML_CUDA=on" .venv/bin/pip install --no-cache-dir -r requirements.txt
.venv/bin/huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models
.venv/bin/python main.py            # writes test-output.txt
.venv/bin/python -I check.py [log.jsonl]
```

No system CUDA toolkit on the GPU box was needed: nvcc/headers/libs came from pip wheels
(`nvidia-cuda-nvcc==13.2.*`, `nvidia-cuda-runtime==13.2.*`, `nvidia-cuda-cccl==13.2.*`,
`nvidia-cuda-crt==13.2.*`, `nvidia-nvvm==13.2.*`, `nvidia-cublas==13.*`), with `lib*.so`
symlinks added in `site-packages/nvidia/cu13/lib` and
`CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=$C/bin/nvcc -DCUDAToolkit_ROOT=$C"`
(`C=.../site-packages/nvidia/cu13`); run with `LD_LIBRARY_PATH=$C/lib`.

## check.py

Rule parser vs `data/LMML_Egypt_TRAIN_DATA/train-output.txt` (must be 13/13), spec edge
cases, strict-JSON parse of every `test-output.txt` line (duplicate keys / NaN rejected),
output == parser data, optional model-vs-rules agreement from the log, and an ambiguity
scan of the test input.

`models/` is not committed.

## Results (RTX 3070 Ti, CPython 3.11.16, llama-cpp-python 0.3.35 CUDA 13.2 sm_86, huggingface-hub 0.26.2)

- train: rules == train-output 13/13; spec/edge cases 6/6
- test-output.txt: 100/100 lines strict JSON, 100/100 equal to the parser; sha256 `7b0b9a89...a2d1`
- model vs rules: 47/100 agree exactly, 47 wrong data (mostly dropped empty fields, kept
  noise such as `page 22`, quoted numbers), 6 invalid JSON (unescaped `\`, stray quotes);
  all 53 repaired to the parser result
- `main.py` end to end on GPU: 31.4 s wall (model load + 100 records), limit 60 s
- Ambiguity scan: no colon-bearing noise, no non-simple keys, no duplicate keys, no
  `+5`/`007`/`-0`/`1.5`/`True`/`null` values; only lines 23 and 72 have a value with a
  second colon (`era`/`city` -> `"cost: $200"`), handled by the first-colon rule.
