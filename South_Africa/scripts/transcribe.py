"""Stage 2: ASR on the cleaned audio (run on GPU box).
 - MMS-1b-all with the isiZulu ('zul') adapter, chunked with CTC word timestamps
 - faster-whisper large-v3 (auto language) with word timestamps
Usage: python transcribe.py cleaned.wav out_dir [tag]
"""
import sys, os, json
import numpy as np, soundfile as sf, torch

inp, out = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "clean"
os.makedirs(out, exist_ok=True)
x, sr = sf.read(inp, dtype="float32")
assert sr == 16000

# ---------- MMS ----------
RUN_MMS = os.environ.get("SKIP_MMS") is None
if RUN_MMS:
  exec(compile(r"""
from transformers import Wav2Vec2ForCTC, AutoProcessor
mid = "facebook/mms-1b-all"
proc = AutoProcessor.from_pretrained(mid, target_lang="zul")
model = Wav2Vec2ForCTC.from_pretrained(mid, target_lang="zul", ignore_mismatched_sizes=True).cuda().eval()
words = []
chunk_words = []
CH, HOP = 20 * sr, 15 * sr  # 20 s windows, 5 s overlap
for st in range(0, len(x), HOP):
    seg = x[st:st + CH]
    if len(seg) < sr // 2:
        break
    inp_ = proc(seg, sampling_rate=sr, return_tensors="pt").input_values.cuda()
    with torch.no_grad():
        logits = model(inp_).logits[0]
    ids = logits.argmax(-1).cpu().numpy()
    fr = len(seg) / sr / len(ids)
    vocab = {v: k for k, v in proc.tokenizer.get_vocab().items()}
    blank = proc.tokenizer.pad_token_id
    cur, cs, prev = "", None, None
    for i, k in enumerate(ids):
        if k != prev and k != blank:
            ch = vocab[int(k)]
            if ch == "|":
                if cur:
                    words.append((st / sr + cs * fr, st / sr + i * fr, cur))
                cur, cs = "", None
            else:
                if cs is None:
                    cs = i
                cur += ch
        prev = k
    if cur:
        words.append((st / sr + cs * fr, st / sr + len(ids) * fr, cur))
    lo = st / sr + (2.5 if st > 0 else 0)
    hi = st / sr + 17.5
    chunk_words.extend(wd for wd in words if lo <= wd[0] < hi)
    words = []
# dedupe overlap: keep words whose start falls in this chunk's non-overlapped core
ded = sorted(chunk_words)
with open(os.path.join(out, f"mms_{tag}.txt"), "w") as fh:
    for a, b, t in ded:
        fh.write(f"{a:8.2f} {b:8.2f} {t}\n")
del model
torch.cuda.empty_cache()
""", "mms", "exec"))

# ---------- Whisper (transformers, torch CUDA) ----------
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model=os.environ.get("WMODEL", "openai/whisper-large-v3"),
               torch_dtype=torch.float16, device="cuda:0")
for lang in [l for l in os.environ.get("WLANGS", "auto,sw").split(",")]:
    kw = {"task": "transcribe", "max_new_tokens": 200}
    if lang != "auto":
        kw["language"] = lang
    r = asr({"raw": x, "sampling_rate": sr}, chunk_length_s=30, batch_size=1,
            return_timestamps=os.environ.get("WTS", "word") if os.environ.get("WTS", "word") == "word" else True, generate_kwargs=kw)
    with open(os.path.join(out, f"whisper_{lang}_{tag}.txt"), "w") as fh:
        for c in r["chunks"]:
            a, b = c["timestamp"]
            fh.write(f"{a if a is not None else -1:8.2f} {b if b is not None else -1:8.2f} {c['text'].strip()}\n")
print("done")
