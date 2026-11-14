"""CTC forced scoring of candidate phrases 'uzosebenzisa ikhodi <W> isicelo' on the region
around the key, for several enhanced variants (MMS zul). Prints greedy decode + ranking.
Usage: python score_candidates.py t0 t1 offset wav [wav ...]   (offset = file start time)"""
import sys, numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
t0, t1, off = map(float, sys.argv[1:4])
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="zul")
m = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="zul", ignore_mismatched_sizes=True).cuda().eval()
cands = ["ulwazi","ulwazo","unwazi","unwazo","undwazo","undwazi","unwabe","unwaso","ulwaso","unwaza","ulwaza"]
cands = ["ulwazi","ulwazo","unwazi","unwazo","undwazo","undwazi","unwabe","unwaso","ulwaso","unwaza","ulwaza"]
tot = {c: 0.0 for c in cands}
for f in sys.argv[4:]:
    a, sr = sf.read(f, dtype="float32")
    x = a[int((t0 - off) * sr):int((t1 - off) * sr)]
    iv = proc(x, sampling_rate=sr, return_tensors="pt").input_values.cuda()
    with torch.no_grad():
        lp = m(iv).logits.log_softmax(-1)
    print(f"== {f.split('/')[-1]}: {proc.decode(lp[0].argmax(-1))}")
    res = {}
    for c in cands:
        lab = torch.tensor([proc.tokenizer(f"uzosebenzisa ikhodi {c} isicelo").input_ids])
        L = torch.nn.functional.ctc_loss(lp.transpose(0, 1).float(), lab.cuda(), [lp.shape[1]], [lab.shape[1]],
                                         blank=proc.tokenizer.pad_token_id, reduction="sum").item()
        res[c] = -L
    best = max(res.values())
    for c in res: tot[c] += res[c] - best
    print("   " + "  ".join(f"{c}:{res[c]-best:.1f}" for c in sorted(res, key=res.get, reverse=True)[:40]))
print("== TOTAL (sum of rel. log-lik)")
print("   " + "  ".join(f"{c}:{tot[c]:.1f}" for c in sorted(tot, key=tot.get, reverse=True)[:80]))
