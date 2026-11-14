"""Stage 1: classical denoising of voice_note.wav.

Findings from inspection: the "crowd" contains two fixed-pitch vuvuzela-like
harmonic stacks (f0 = 217.5 Hz and 233.3 Hz, harmonics up to 8 kHz) that switch
on/off, plus broadband crowd noise. We
  1. comb-notch both harmonic series in a high-resolution STFT,
  2. apply a speech bandpass (80 Hz - 7.5 kHz),
  3. apply non-stationary spectral gating (noisereduce).
Usage: python denoise.py in.wav out_dir
"""
import sys, os
import numpy as np
import scipy.io.wavfile as w
import scipy.signal as s

inp = sys.argv[1] if len(sys.argv) > 1 else "data/voice_note.wav"
out = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(out, exist_ok=True)
sr, x = w.read(inp)
x = x.astype(np.float32) / 32768


def estimate_f0s(x, sr):
    """Refine the two tonal fundamentals from the long-term spectrum."""
    n = 1 << 20
    F = np.zeros(n // 2 + 1)
    for i in range(0, len(x) - n, n // 2):
        F += np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / sr)
    res = []
    for guess in (217.5, 233.3):
        # use harmonic-summed refinement
        cands = np.arange(guess - 1.5, guess + 1.5, 0.005)
        score = [sum(np.interp(k * c, f, F) for k in range(1, 20)) for c in cands]
        res.append(float(cands[int(np.argmax(score))]))
    return res


f0s = estimate_f0s(x, sr)
print("tonal fundamentals:", f0s)

# 1. comb notch in STFT domain
nper = 4096
f, t, S = s.stft(x, sr, nperseg=nper, noverlap=nper * 3 // 4)
mask = np.ones(len(f))
for f0 in f0s:
    for k in range(1, int(8000 / f0) + 1):
        mask[np.abs(f - k * f0) < 9.0] = 0.0
S *= mask[:, None]
_, y = s.istft(S, sr, nperseg=nper, noverlap=nper * 3 // 4)
y = y[: len(x)]

# 2. bandpass
sos = s.butter(6, [80, 7500], btype="band", fs=sr, output="sos")
y = s.sosfiltfilt(sos, y)
w.write(os.path.join(out, "stage1_notched.wav"), sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))

# 3. spectral gating
import noisereduce as nr
z = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.9, n_fft=1024)
z = z / (np.abs(z).max() + 1e-9) * 0.9
w.write(os.path.join(out, "cleaned.wav"), sr, (z * 32767).astype(np.int16))
print("wrote", out)
