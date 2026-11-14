"""Second-opinion ASR: SeamlessM4T v2 large, speech->text in isiZulu (tgt_lang='zul').
Usage: python seamless_asr.py t0 t1 offset wav [wav ...]"""
import sys, numpy as np, soundfile as sf, torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
t0, t1, off = map(float, sys.argv[1:4])
dev = "cuda" if torch.cuda.mem_get_info()[0] > 5.5e9 else "cpu"
proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
m = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large",
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
for f in sys.argv[4:]:
    a, sr = sf.read(f, dtype="float32")
    x = a[int((t0 - off) * sr):int((t1 - off) * sr)]
    inp = proc(audio=x, sampling_rate=sr, return_tensors="pt").to(dev)
    if dev == "cuda": inp["input_features"] = inp["input_features"].half()
    for tl in ("zul", "eng"):
        with torch.no_grad():
            out = m.generate(**inp, tgt_lang=tl, num_beams=5, max_new_tokens=100)
        print(f"{f.split('/')[-1]} [{tl}]: {proc.decode(out[0], skip_special_tokens=True)}")
