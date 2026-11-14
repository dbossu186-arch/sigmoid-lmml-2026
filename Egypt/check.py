#!/usr/bin/env python3
"""Local harness.

  python -I check.py [model_log.jsonl]

1. Rule parser vs train-output (data/LMML_Egypt_TRAIN_DATA/), must be 100%.
2. Spec examples / edge cases.
3. test-output.txt: line count == test-input, every line strict JSON object (no
   duplicate keys, no NaN), and data == rule parser on the matching input line.
4. Optional: model-vs-rules agreement from a log written by main.py (EGYPT_LOG).
5. Lists ambiguity signals in test-input (non-simple keys with ':', odd numbers, dup keys).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main as M  # noqa: E402

TRAIN = os.path.join(HERE, "data", "LMML_Egypt_TRAIN_DATA")
ok = True


def strict(s):
    def hook(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(pairs)

    def bad(c):
        raise ValueError(c)
    o = json.loads(s, object_pairs_hook=hook, parse_constant=bad)
    if not isinstance(o, dict):
        raise ValueError("not object")
    return o


def report(name, good, total):
    global ok
    ok &= good == total
    print("%-40s %d/%d %s" % (name, good, total, "OK" if good == total else "FAIL"))


# 1. train
tin = M.read_lines(os.path.join(TRAIN, "train-input.txt"))
tout = M.read_lines(os.path.join(TRAIN, "train-output.txt"))
assert len(tin) == len(tout)
g = 0
for i, (a, b) in enumerate(zip(tin, tout)):
    d = M.parse_record(a)[0]
    if M.same_data(d, strict(b)) and M.same_data(strict(M.dumps(d)), strict(b)):
        g += 1
    else:
        print("  train mismatch line", i + 1, M.dumps(d), "!=", b)
report("train: rules == train-output", g, len(tin))

# 2. spec / edge cases
cases = [
    ('page 22| king :Menes| title: ruled with "the Aten" |[torn]| alive :true | Chapter IX',
     {"king": "Menes", "title": 'ruled with "the Aten"', "alive": True}),
    ("ratio: 3:1", {"ratio": "3:1"}),
    ("a: 40|b: 0|c: -3|d:|e: true|f: false", {"a": 40, "b": 0, "c": -3, "d": None, "e": True, "f": False}),
    ("a: True|b: TRUE|c: null|d: 1.5|e: 1e3|f: +5|g: 007|h: -0",
     {"a": "True", "b": "TRUE", "c": "null", "d": "1.5", "e": "1e3", "f": "+5", "g": "007", "h": "-0"}),
    ("Fig. 3: a map | Plate II | x: a\\b | - 45 - | 22 |", {"x": "a\\b"}),
    ("\tk\t:\tv \r\n", {"k": "v"}),
]
g = 0
for inp, exp in cases:
    d = M.parse_record(inp)[0]
    if M.same_data(d, exp) and M.same_data(strict(M.dumps(d)), exp):
        g += 1
    else:
        print("  case fail", repr(inp), d)
report("spec examples / edge cases", g, len(cases))

# 3. test output
tin = M.read_lines(os.path.join(HERE, "test-input.txt"))
outp = os.path.join(HERE, "test-output.txt")
tout = M.read_lines(outp) if os.path.exists(outp) else []
report("test-output line count", len(tout) if len(tout) == len(tin) else 0, len(tin))
g_json = g_data = 0
for i, (a, b) in enumerate(zip(tin, tout)):
    try:
        o = strict(b)
        g_json += 1
    except Exception as e:
        print("  invalid JSON line", i + 1, e)
        continue
    if M.same_data(o, M.parse_record(a)[0]):
        g_data += 1
    else:
        print("  data mismatch line", i + 1)
raw = open(outp, "rb").read() if os.path.exists(outp) else b""
report("test-output strict JSON", g_json, len(tin))
report("test-output == rule parser", g_data, len(tin))
print("  output bytes: utf-8 ok=%s, CR present=%s" % (bool(raw.decode("utf-8")), b"\r" in raw))

# 4. model log
if len(sys.argv) > 1:
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print("model vs rules:", dict(c), "agree %d/%d" % (c["agree"], len(rows)))
    for r in rows:
        if r["status"] != "agree":
            print("  line %d %s\n    model: %s\n    rules: %s" % (r["line"], r["status"], r["model"], r["rules"]))

# 5. ambiguity scan of test input
print("ambiguity scan of test-input:")
n = 0
for i, line in enumerate(tin):
    keys = []
    for seg in line.split("|"):
        s = seg.strip()
        if ":" in s:
            k, v = s.split(":", 1)
            k, v = k.strip(), v.strip()
            if not M.KEY_RE.match(k):
                print("  line %d: colon segment dropped as noise: %r" % (i + 1, s)); n += 1
                continue
            keys.append(k)
            if re.fullmatch(r"[+-]?\d+(\.\d*)?([eE][+-]?\d+)?", v) and not M.INT_RE.match(v) or v in ("-0",):
                print("  line %d: numeric-looking string kept as string: %r" % (i + 1, v)); n += 1
            if v.lower() in ("true", "false", "null", "none") and v not in ("true", "false"):
                print("  line %d: bool/null-looking string: %r" % (i + 1, v)); n += 1
            if ":" in v:
                print("  line %d: value contains ':' (first-colon split): %r -> %r" % (i + 1, k, v)); n += 1
        elif s and re.search(r"[A-Za-z]", s) and not re.fullmatch(
                r"(page \d+|p\. \d+|Chapter [IVXLC\d]+|Plate [IVXLC\d]+|Fig\.? \d+|end of page|see above|scanned \d+|"
                r"\[[a-z ]+\]|\([a-z ]+\))", s):
            print("  line %d: unusual noise segment: %r" % (i + 1, s)); n += 1
    if len(keys) != len(set(keys)):
        print("  line %d: DUPLICATE KEYS %s" % (i + 1, keys)); n += 1
print("  %d flags" % n)
print("ALL OK" if ok else "SOMETHING FAILED")
sys.exit(0 if ok else 1)
