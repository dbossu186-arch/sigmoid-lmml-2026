"""Stage 1b: neural vocal separation with Demucs htdemucs (GPU) on the notched audio.
Usage: python separate.py in.wav out.wav"""
import sys, numpy as np, soundfile as sf, torch, librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model
x, sr = sf.read(sys.argv[1], dtype="float32")
m = get_model("htdemucs_ft" if len(sys.argv) < 4 else sys.argv[3]).cuda().eval()
y = librosa.resample(x, orig_sr=sr, target_sr=m.samplerate)
wav = torch.tensor(np.stack([y, y]))[None].cuda()
with torch.no_grad():
    src = apply_model(m, wav, shifts=1, split=True, overlap=0.25, segment=7)[0]
v = src[m.sources.index("vocals")].mean(0).cpu().numpy()
v = librosa.resample(v, orig_sr=m.samplerate, target_sr=16000)
v = v / (np.abs(v).max() + 1e-9) * 0.9
sf.write(sys.argv[2], v, 16000, subtype="PCM_16")
print("ok", m.sources)
