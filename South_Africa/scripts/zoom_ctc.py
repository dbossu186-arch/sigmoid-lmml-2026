"""Frame-level CTC inspection of a region (MMS zul): greedy + top-3 per non-blank frame.
Usage: python zoom_ctc.py wav t0 t1 [wav2 ...]"""
import sys, numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
t0, t1 = float(sys.argv[2]), float(sys.argv[3])
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="zul")
m = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="zul", ignore_mismatched_sizes=True).cuda().eval()
voc = {v: k for k, v in proc.tokenizer.get_vocab().items()}
for f in [sys.argv[1]] + sys.argv[4:]:
    a, sr = sf.read(f, dtype="float32")
    for pad in (0.0, 0.3, 0.7):
        x = a[int((t0 - pad) * sr):int((t1 + pad) * sr)]
        iv = proc(x, sampling_rate=sr, return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            lp = m(iv).logits[0].log_softmax(-1).cpu().numpy()
        print(f"== {f} pad {pad}: {proc.decode(lp.argmax(-1))}")
        if pad == 0.3:
            fr = len(x) / sr / len(lp)
            for i, row in enumerate(lp):
                top = np.argsort(row)[::-1][:3]
                if top[0] != proc.tokenizer.pad_token_id or row[top[0]] < -0.5:
                    print(f"   {t0-pad+i*fr:7.2f} " + "  ".join(f"{voc[k]}:{np.exp(row[k]):.2f}" for k in top))
