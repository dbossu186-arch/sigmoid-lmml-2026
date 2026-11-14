"""Stage 2b: MMS (zul adapter) on VAD-segmented utterances, using pause detection
on the separated vocals. Also dumps per-segment greedy transcript.
Usage: python mms_vad.py vad_source.wav asr_source.wav out.txt"""
import sys, numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
v, sr = sf.read(sys.argv[1], dtype="float32")
a, _ = sf.read(sys.argv[2], dtype="float32")
hop = 160
e = np.array([np.sqrt(np.mean(v[i:i+400]**2)) for i in range(0, len(v)-400, hop)])
e_db = 20*np.log10(e+1e-6)
thr = np.percentile(e_db, 30) + 6
act = e_db > thr
# close gaps < 0.35 s, segments split on pauses >= 0.35 s
segs = []; i = 0; n = len(act)
while i < n:
    if act[i]:
        j = i
        last = i
        while j < n and (j - last) < 35:
            if act[j]: last = j
            j += 1
        segs.append((i, last+1)); i = j
    else:
        i += 1
segs = [(max(0, s*hop/sr-0.15), min(len(a)/sr, e_*hop/sr+0.15)) for s, e_ in segs if (e_-s)*hop/sr > 0.25]
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="zul")
m = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="zul", ignore_mismatched_sizes=True).cuda().eval()
with open(sys.argv[3], "w") as fh:
    for s0, s1 in segs:
        seg = a[int(s0*sr):int(s1*sr)]
        # split very long segments
        outs = []
        for k in range(0, len(seg), 25*sr):
            x = seg[k:k+25*sr]
            if len(x) < 4000: continue
            iv = proc(x, sampling_rate=sr, return_tensors="pt").input_values.cuda()
            with torch.no_grad():
                ids = m(iv).logits[0].argmax(-1)
            outs.append(proc.decode(ids))
        fh.write(f"{s0:8.2f} {s1:8.2f} {' '.join(outs)}\n")
print(len(segs), "segments")
