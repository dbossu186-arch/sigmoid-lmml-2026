# Sigmoid LMML 2026: Japan, "City of Subtitles" (Hard, 300 pts): working notes

## The task
- **Story:** the account 字幕の街 ("City of Subtitles") posts city photos with the "Subtitle Effect": warmer colours, added grain, and a Japanese caption bar.
- **Photographers:** six regulars, Yakin, Amayadori, Shuuden, Jihanki, Rojiura and Toorisugari.
- **Credited vs anonymous:** weeks 1–7 are credited, with 18 photos per person. Weeks 8–12 are 145 anonymous photos.
- **Rumour:** Toorisugari runs the account and plays along with their own phone.
- **Clue 1:** "Yakin even bought the same phone model."
- **Clue 2 (week 13):** *"The photos of a simple passer-by might know more than they show. Who knows, maybe an unknown photographer identity hides there."*
- **Goal:** attribute each anonymous photo from the camera traces in its pixels, find Toorisugari's photos, and "discover the secret".
- **Flag:** `SIGMOID_<WORD>`, the hidden word in uppercase.
- **Submit:** the flag plus a URL to code and a short write-up showing (a) how the photos were attributed and (b) how the word appeared. Wrong flag or no write-up scores 0.
- **Hints are locked and cost points.** Don't use them.
- **Data (public Google Drive, 79 MB):** https://drive.google.com/drive/folders/1dKvKxO5BTcpjR9QSumq5pAfeRYXYYS92
  - `credited/<name>/` holds 18 JPEGs per person.
  - `anonymous/` holds 145 JPEGs.
  - All images are 1280×960 with no EXIF. File names look like `wWW_NNN.jpg`, where WW is the week and NNN is a meaningless counter.
  - `baseline_solution.py` is included.
  - If gdown is rate-limited, list the file IDs with `gdown.download_folder(id=..., skip_download=True)`, then fetch each with `curl -sL "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t"`.

## What we've learned (about 2.5 h of work, two independent attempts)

**Method that works for attribution:**
- Compute a PRNU wavelet residual (Mihcak denoiser, sigma ≈ 3).
- Mask out the caption bars and translucent boxes before computing it.
- Build an MLE fingerprint per camera and score matches with PCE.
- Run transductive rounds, adding confident anonymous photos to the fingerprints.
- Leave-one-out accuracy on credited photos is about 76%. Night (Yakin) and rain (Amayadori) photos are weakest.

**Orientation is the big trap:**
- About 60 of the 145 anonymous photos, plus some credited ones, from every camera, match best after a **180° rotation**, because people hold the phone either way round in landscape.
- Compute PCE in both orientations, and rotate before averaging when you build a fingerprint.
- We first thought we'd found a "7th camera X". It turned out to be Toorisugari's phone rotated 180°: PCE 427, against a ±30–50 null for other pairings.

**Other observations:**
- The grain and PRNU are monochrome, and every JPEG quantisation table is identical, so there's no easy camera-model cue.
- Yakin has the **same phone model** as Toorisugari, so their fingerprints share a model-level component. Removing that component should separate them.

**Tried, found nothing:**
- rendering the fingerprints as images, t-maps, hot pixels, column profiles
- the week × counter grid

**Not yet tried properly:**
- a real 7th identity among the leftover low-confidence photos, once attribution is orientation-aware
- a watermark or pattern embedded in Toorisugari's photos, e.g. the residual of only their photos, averaged with orientation fixed, after subtracting the model component they share with Yakin
- the caption text of Toorisugari's photos
- per-week ordering or counts of Toorisugari's photos mapped to letters

## Code
The scripts in this folder: `prnu_lib.py`, `stage1_residuals.py`, `stage2_attribute.py`, `orient_attr.py`, `joint_em.py`, `openset.py` and the others.
