#!/usr/bin/env python3
"""Egypt - Always Valid, No Excuses.

Reads test-input.txt (beside this file), writes test-output.txt (beside this file):
one strict JSON object per input line.

Pipeline
  1. A deterministic parser (`parse_record`) implements the conversion spec exactly.
     Its result is written to test-output.txt immediately, so a valid file exists
     even if the model step is slow or crashes.
  2. The local model Qwen3-1.7B (Q8_0 GGUF, llama-cpp-python, n_gpu_layers=-1) is
     prompted (few-shot, greedy) to convert every record itself.
  3. Every model line is held to the format: it must parse as strict JSON and decode
     to exactly the parser's data. Agreeing lines are kept; any line that is invalid
     or differs is repaired to the parser's result. The file is then rewritten.
     So the final output is always the spec-exact data, whatever the model does.

Offline: the model is loaded from models/Qwen3-1.7B-Q8_0.gguf; nothing is downloaded.
Env vars: EGYPT_MODEL_BUDGET (seconds of wall time allowed for the model phase,
default 45), EGYPT_NO_MODEL=1 (rules only), EGYPT_LOG (path for a per-line JSONL log).
"""
import json
import os
import re
import sys
import time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "test-input.txt")
OUT_PATH = os.path.join(HERE, "test-output.txt")
MODEL_PATH = os.path.join(HERE, "models", "Qwen3-1.7B-Q8_0.gguf")

# ---------------------------------------------------------------- deterministic spec
# A real field is a segment shaped "key: value" whose key (trimmed) is a simple word.
KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# Whole number: optional minus, digits, no leading zeros (40, 0, -3). "+5", "007",
# "-0", "1.5", "1e3" stay strings (none of these occur in the test input).
INT_RE = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")


def convert_value(v):
    if v == "":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    if INT_RE.match(v) and v != "-0":
        return int(v)
    return v


def parse_record(line):
    """Return (dict of fields in order, list of dropped noise segments)."""
    line = line.rstrip("\r\n")
    fields, noise = {}, []
    for seg in line.split("|"):
        if ":" not in seg:
            if seg.strip():
                noise.append(seg.strip())
            continue
        key, value = seg.split(":", 1)  # only the first ':' separates key from value
        key, value = key.strip(), value.strip()
        if not KEY_RE.match(key):
            noise.append(seg.strip())
            continue
        fields[key] = convert_value(value)
    return fields, noise


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def strict_loads(s):
    """json.loads that rejects NaN/Infinity and requires a top-level object."""
    def bad_const(c):
        raise ValueError("non-standard constant " + c)
    obj = json.loads(s, parse_constant=bad_const)
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def same_data(a, b):
    """Exact equality of decoded JSON, with bool/int kept distinct (True != 1)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_data(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_data(x, y) for x, y in zip(a, b))
    return a == b


def read_lines(path):
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    if text.startswith("\ufeff"):
        text = text[1:]
    lines = re.split(r"\r\n|\n|\r", text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def write_lines(path, lines):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for l in lines:
            f.write(l + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- model (the oracle)
SYSTEM = """You convert one OCR'd museum record card into one line of strict JSON.
The record is segments separated by |. Rules:
- A segment is a real field only if it is shaped key: value with a simple word key. Drop scan noise: page numbers (page 22, p. 128, 22), headers (Chapter IX), footers (- 45 -, end of page), bracketed marks ([torn], [illegible], (margin note), (faded)), figure labels (Fig 3, Plate II), garble (###, ~~~, * * *), stray notes (see above, scanned 1993).
- Trim spaces around every key and value. Only the FIRST colon separates key from value; later colons belong to the value.
- Empty value -> null. Exactly true / false -> JSON booleans. Whole numbers like 40, 0, -3 -> JSON numbers.
- Anything else -> JSON string, copied exactly, with " escaped as \\" and \\ escaped as \\\\.
- Every segment that has a word key and a colon is a field, including the last one, even when its value is empty.
- Numbers are written without quotes: "rank": -18, "age": 0, "tomb": 2103.
- Keep field order. Add nothing. Output only the JSON object on one line."""

EXAMPLES = [
    ("page 22| king :Menes| title: ruled with \"the Aten\" |[torn]| alive :true | Chapter IX",
     {"king": "Menes", "title": "ruled with \"the Aten\"", "alive": True}),
    ("god: Thoth | title: \"lord of the moon\" | scroll: C:\\papyri\\thoth | - 12 -",
     {"god": "Thoth", "title": "\"lord of the moon\"", "scroll": "C:\\papyri\\thoth"}),
    (" reign: true  |  (faded)  |  ###  |  era : Seti I  |  rank : -18   |  height:    |  god:-57  |  page 7  |  city: Amun",
     {"reign": True, "era": "Seti I", "rank": -18, "height": None, "god": -57, "city": "Amun"}),
    ("ratio: 2:1 | width: 230 | epithet : cost: $200 | material: limestone & granite | Plate VII | 128",
     {"ratio": "2:1", "width": 230, "epithet": "cost: $200", "material": "limestone & granite"}),
    ("[illegible] | pharaoh : Tutankhamun | age : 0 | tomb: KV62 | gold: false | notes: see p.45 (faded) | end of page",
     {"pharaoh": "Tutankhamun", "age": 0, "tomb": "KV62", "gold": False, "notes": "see p.45 (faded)"}),
]


def user_msg(rec):
    """Show the raw record, then its |-separated segments one per line (outer spaces
    stripped) so the small model does not lose fields at the end."""
    segs = rec.rstrip("\r\n").split("|")
    return "Record: " + rec.rstrip("\r\n") + "\nSegments:\n" + "\n".join(
        "%d. %s" % (i + 1, s.strip()) for i, s in enumerate(segs))


def build_prefix():
    p = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    for rec, obj in EXAMPLES:
        p += "<|im_start|>user\n" + user_msg(rec) + "<|im_end|>\n"
        p += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + json.dumps(obj, ensure_ascii=False) + "<|im_end|>\n"
    return p


def run_model(lines, deadline):
    """Return list of model output strings (None where not run)."""
    from llama_cpp import Llama
    try:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
        draft = LlamaPromptLookupDecoding(num_pred_tokens=10)
    except Exception:  # pragma: no cover
        draft = None
    # logits_all=True is required with a draft model in llama-cpp-python 0.3.35: it
    # forces logits_all internally but sizes the score buffer from the argument.
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=3072, n_batch=1024,
                seed=0, verbose=False, draft_model=draft, logits_all=draft is not None)
    prefix = build_prefix()
    outs = [None] * len(lines)
    for i, line in enumerate(lines):
        if time.time() > deadline:
            break
        prompt = prefix + "<|im_start|>user\n" + user_msg(line) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        r = llm.create_completion(prompt, max_tokens=400, temperature=0.0, top_k=1,
                                  repeat_penalty=1.0, stop=["<|im_end|>", "\n"])
        outs[i] = r["choices"][0]["text"].strip()
    return outs


def main():
    lines = read_lines(IN_PATH)
    parsed = [parse_record(l)[0] for l in lines]
    rule_lines = [dumps(d) for d in parsed]
    for s, d in zip(rule_lines, parsed):  # self-check of the serializer
        assert same_data(strict_loads(s), d)
    write_lines(OUT_PATH, rule_lines)  # safe output exists from here on

    model_outs = [None] * len(lines)
    err = None
    if os.environ.get("EGYPT_NO_MODEL") != "1":
        budget = float(os.environ.get("EGYPT_MODEL_BUDGET", "45"))
        try:
            model_outs = run_model(lines, T0 + budget)
        except Exception as e:  # model failure must never cost a record
            err = repr(e)

    final, stats, log = [], {"agree": 0, "invalid_json": 0, "wrong_data": 0, "not_run": 0}, []
    for i, (line, d) in enumerate(zip(lines, parsed)):
        m = model_outs[i]
        status = "not_run"
        if m is not None:
            try:
                md = strict_loads(m)
                status = "agree" if same_data(md, d) else "wrong_data"
            except Exception:
                status = "invalid_json"
        stats[status] += 1
        # keep the model's line only when it is exactly right; otherwise repair
        final.append(dumps(strict_loads(m)) if status == "agree" else rule_lines[i])
        log.append({"line": i + 1, "status": status, "model": m, "rules": rule_lines[i]})
    for s, d in zip(final, parsed):
        assert same_data(strict_loads(s), d)
    write_lines(OUT_PATH, final)

    if os.environ.get("EGYPT_LOG"):
        with open(os.environ["EGYPT_LOG"], "w", encoding="utf-8") as f:
            for row in log:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    msg = "records=%d model: %s repaired=%d elapsed=%.1fs" % (
        len(lines), stats, len(lines) - stats["agree"], time.time() - T0)
    if err:
        msg += " model_error=" + err
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
