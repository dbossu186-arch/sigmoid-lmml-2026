# Japan — "City of Subtitles" (Hard, Computer Vision) — Sigmoid LMML 2026

## Storyline (key wording)
Jimaku Kōka (字幕効果), the "Subtitle Effect": warm the colours, add grain, and set a Japanese caption under a dull photo. The anonymous account 字幕の街 (Jimaku no Machi, "City of Subtitles") ran a weekly game: followers photographed a mediocre city corner, and the account applied the effect and posted the photos. Six regulars: 夜勤 Yakin (night shift), 雨宿り Amayadori (sheltering from the rain), 終電 Shuuden (last train), 自販機 Jihanki (vending machine), 路地裏 Rojiura (back alley), 通りすがり Toorisugari (just passing by). Weeks 1–7 were posted with names; from week 8 without. Rumour had it Toorisugari ran the account, playing along with their own phone. "Yakin even bought the same phone model, hoping that whatever made those pictures would rub off." In week 13 the account posted only: "The photos of a simple passer-by might know more than they show. Who knows, maybe an unknown photographer identity hides there."

## Objective
Work out who took each nameless photo of weeks 8–12 from what the cameras left in the pixels. Find the ones taken by Toorisugari (toorisugari), and discover the secret.

## Flag
SIGMOID_<WORD>, where <WORD> is the hidden word in uppercase.

## Input
JPEGs, all 1280×960, all with the Subtitle Effect, no metadata:
- credited/: weeks 1–7, one subfolder per person (toorisugari, yakin, amayadori, shuuden, jihanki, rojiura), 18 photos each.
- anonymous/: 145 nameless photos of weeks 8–12.
- File names are wWW_NNN.jpg: WW is the posting week, NNN a meaningless counter.
- baseline_solution.py: loads the photos and makes a naive first comparison.

Dataset: https://drive.google.com/drive/folders/1dKvKxO5BTcpjR9QSumq5pAfeRYXYYS92 (ID 1dKvKxO5BTcpjR9QSumq5pAfeRYXYYS92)

## Expected output / scoring
Submit the flag plus a URL to code and a short write-up showing how the photos were attributed and how the word appeared. Correct flag with write-up: 300 points. Wrong flag, or no write-up: 0. 4 hints, locked (penalty).
