"""Sigmoid Nigeria - The Gangan Never Lies: spectrogram steganography.
The 'test tone' is white noise with a band of extra energy (~3.0-3.6 kHz) whose
time-frequency shape spells a word. We render the spectrogram, threshold the band
and save it as figure; the word is read off the image."""
import sys, numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'data/challenge.wav'
sr, x = wavfile.read(path)
f, t, S = spectrogram(x.astype(float), sr, nperseg=1024, noverlap=896)
D = 10*np.log10(S+1e-12)
band = (f > 2800) & (f < 3800)
# smooth & normalise against the noise floor, then threshold -> binary text mask
from scipy.ndimage import uniform_filter
Dn = uniform_filter(D, size=(3, 5)) - np.median(D[~band], axis=0)
mask = Dn[band] > np.percentile(Dn[band], 70)

fig, ax = plt.subplots(2, 1, figsize=(16, 7))
ax[0].pcolormesh(t, f, D, shading='auto', cmap='magma')
ax[0].set_title('challenge.wav spectrogram (full band)'); ax[0].set_ylabel('Hz')
ax[1].imshow(mask, origin='lower', aspect='auto', cmap='gray',
             extent=[t[0], t[-1], f[band][0], f[band][-1]])
ax[1].set_title('2.8-3.8 kHz band, thresholded'); ax[1].set_xlabel('s'); ax[1].set_ylabel('Hz')
plt.tight_layout(); plt.savefig('spectrogram_flag.png', dpi=110)

WORD = 'TALKINGDRUM'   # read from spectrogram_flag.png
print(f'SIGMOID_{WORD}')
