"""Stage 4 (local, light): cut key clips and draw the key spectrogram.
Usage: python scripts/make_outputs.py   (expects out/demucs_notched.wav, out/cleaned.wav)"""
import numpy as np, scipy.io.wavfile as w, scipy.signal as s, shutil, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0, T1 = 437.2, 441.2
shutil.copy("out/demucs_notched.wav", "cleaned.wav")
sr, d = w.read("out/demucs_notched.wav"); _, c = w.read("out/cleaned.wav"); _, r = w.read("data/voice_note.wav")
w.write("key_clip.wav", sr, d[int(T0*sr):int(T1*sr)])
w.write("key_clip_classical.wav", sr, c[int(T0*sr):int(T1*sr)])
w.write("key_clip_raw.wav", sr, r[int(T0*sr):int(T1*sr)])
w.write("key_sentence_clean.wav", sr, d[int(429.8*sr):int(443.2*sr)])
fig, ax = plt.subplots(2, 1, figsize=(16, 9))
for a, (x, ttl) in zip(ax, [(r, "raw voice_note.wav"), (d, "cleaned (comb notch + Demucs htdemucs_ft vocals)")]):
    seg = x[int(T0*sr):int(T1*sr)].astype(float) / 32768
    f, t, S = s.stft(seg, sr, nperseg=512, noverlap=448); D = 10*np.log10(np.abs(S)**2 + 1e-12)
    a.pcolormesh(t + T0, f, D, vmin=D.max()-65, vmax=D.max(), cmap="magma"); a.set_title(ttl); a.set_xlim(T0, T1)
    for tt, lab in [(437.55, "uzosebenzisa"), (438.65, "ikhodi"), (439.8, "ULWAZI")]:
        a.axvline(tt, color="c", lw=1); a.text(tt + 0.02, 7400, lab, color="c", fontsize=12)
ax[1].set_xlabel("time in voice_note.wav (s)"); plt.tight_layout(); plt.savefig("spectrogram_key.png", dpi=80)
