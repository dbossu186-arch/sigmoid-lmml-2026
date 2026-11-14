# Sigmoid South Africa: "Ngiyakuzwa?"

**Answer:** the isiZulu word spoken right after "ikhodi" (see below, transcript.txt and key_clip.wav).

- "ikhodi": **438.65 to 439.30 s** (7:18.7) in `data/voice_note.wav`
- Keyword: **439.80 to 440.50 s** (7:19.8), with a short pause on each side
- Sentence (cleaned audio, about 429.8 to 443 s): *"irisidi lokwamukela isicelo … uzosebenzisa **ikhodi** **ulwazi** isicelo …"*
- Listen: `key_clip.wav` (437.2 to 441.2 s, cleaned), `key_clip_raw.wav` (same span, raw), `key_sentence_clean.wav` (the whole sentence)

## What the noise is
Spectrogram (`spectrogram_overview.png`): the "crowd" is mostly two fixed-pitch vuvuzela-like harmonic stacks that switch on and off. Their fundamentals are **217.4 Hz and 233.1 Hz**, with harmonics up to 8 kHz. Broadband crowd noise sits under them. The voice is visible mostly through its sibilants at 5 to 7 kHz.

## Pipeline (heavy steps on a GPU machine)
1. `scripts/denoise.py`: STFT comb notch (4096-point frames, ±9 Hz around every harmonic of both fundamentals), then an 80 Hz to 7.5 kHz bandpass, then noisereduce. Writes `out/stage1_notched.wav` and `out/cleaned.wav` (the classical version).
2. `scripts/separate.py`: Demucs `htdemucs_ft` vocals stem from the notched audio, giving `cleaned.wav` (the main result). For the key region it was also run with htdemucs and mdx_extra, on both raw and notched input.
3. `scripts/mms_vad.py`: `facebook/mms-1b-all` with the isiZulu (`zul`) adapter, run on utterances split at pauses. Output: `transcript.txt`. Plain grep finds no exact "ikhodi". The only plausible match is `ikhudi` / `ikhuti` / `ikhuji` at 438.7 s, in "uzosebenzisa ikhudi unwazo".
4. Confirmation:
   - SeamlessM4T v2 large (isiZulu output) on the separated excerpt writes **"ikhodi"** outright: "…kufanele sisebenzise ikhodi" and "Wenela uzosebenzisa ikhodi, ulwazi, isicelo sethu."
   - `scripts/zoom_ctc.py` shows MMS's top letters frame by frame.
   - `scripts/score_candidates*.py` scores candidate words with CTC across 9 enhancement variants.
5. Whisper large-v3 and turbo do not support isiZulu. They only produced hallucinated loops, so they are not used as evidence.

## Why ULWAZI, and what the alternatives are
The sound of the word is clear: **u + (l or n) + w + a + z/s + a final vowel that is almost silent**.
- In the zoom (`spectrogram_word_zoom.png`), 439.80 to 440.00 s is voiced and low; 440.05 to 440.25 s is an /a/ (F1 about 700 Hz, F2 about 1500 Hz); 440.27 to 440.45 s is a strong sibilant at 5 to 7.5 kHz. The final vowel after the sibilant is nearly voiceless. That is normal for a high vowel /i/ after /z/.
- MMS greedy outputs (no language model) across variants: `ulwazo` ×3, `unwazo` ×2, `ulwaso`, `unywazi`, `undwazi`.
- SeamlessM4T (which has a language model): `ulwazi`, `uNwazi`, `unwazo`.
- **ulwazi** is the only standard isiZulu word consistent with all of these.

Alternative spellings, from most to least likely:
- ULWAZI (chosen)
- UNWAZO (top in the MMS sum of scores, but not a dictionary word)
- ULWAZO
- UNWAZI

Real words that fit badly: unwabu, ingwazi, imbazo, uzwano.

## Files
- `solve.py`: the driver; runs everything over ssh
- `scripts/`: all the stage scripts
- `cleaned.wav`: the full cleaned audio
- `key_clip*.wav`, `key_sentence_clean.wav`: clips of the key region
- `spectrogram_key.png`, `spectrogram_key_variants.png`, `spectrogram_word_zoom.png`, `spectrogram_overview.png`
- `transcript.txt`
- `out/`: intermediate files and all the MMS transcripts
