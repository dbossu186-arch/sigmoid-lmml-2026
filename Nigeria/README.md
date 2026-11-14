# Nigeria — The Gángan Never Lies

**Flag: `SIGMOID_<redacted>`**

## What was hidden
The "test tone" is broadband white noise, plus extra energy in a narrow band (about 3.0–3.6 kHz) shaped as block capitals. On a spectrogram (time on x, frequency on y) the shape reads **TALKINGDRUM**, upright and left to right, from about 0.5 s to 9.6 s. When you just listen, it sounds like a slight buzz on top of the hiss.

## How it was found
1. Header check: a plain 44-byte RIFF/WAVE header (`fmt ` and `data` chunks only). There are no LIST/INFO chunks and no bytes after the data chunk.
2. LSB check: the bit-0, bit-1 and bit-2 planes, read MSB-first and LSB-first, show no ASCII text or file signatures. They look random.
3. Spectrogram (STFT, nperseg=1024): the word stands out right away. `solve.py` zooms in on 2.8–3.8 kHz and thresholds that band, which gives the clean mask in `spectrogram_flag.png`.

## Files
- `solve.py`: run `python -I solve.py data/challenge.wav`. It renders the figure and prints the flag. The word itself was read from the image by eye, not by OCR.
- `spectrogram_flag.png`: the full spectrogram and the thresholded text band.
