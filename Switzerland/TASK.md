# SWITZERLAND: The Weak Layer (Easy)

## Storyline
High above Davos, the WSL Institute for Snow and Avalanche Research (SLF) has been digging snow pits for decades. An observer climbs to a slope, cuts a clean vertical face into the snowpack, and works down through it layer by layer: grain sizes under a magnifier, hardness by hand, the temperature profile from surface to ground. Every winter those observations feed the national avalanche bulletin.

The most dangerous thing in a snowpack is rarely visible from the surface. It is the weak layer: a band of faceted crystals buried under a slab, formed weeks earlier when bitter cold at the surface met warmth held in the ground, and rounded grains grew into fragile cups. The slab above it can look perfectly solid right up until it isn't.

An observer has just retired after twenty-five winters on the mountain, leaving behind an archive of pit records and a note for whoever inherits them:

> "You will want to know which slopes released. That was never the interesting part. Fit the thing a forecaster fits, then stop looking at what it predicts and look at how it decided. The snowpack asks its questions in a fixed order. Read them in that order and you will have my word."

## Objective
You are given a table of snow-pit observations, each labelled with whether the slope released. Train a decision tree classifier on it.

The answer is not in the tree's predictions. A well-fitted tree here will be close to perfect, and that tells you nothing. The answer is encoded in the structure of the fitted tree. Export the fitted tree, read its structure, and recover the hidden word.

Fit with the library's default hyperparameters and an integer random_state. The answer depends on the shape of the tree, so constraining the fit by capping depth, pruning or limiting features will change that shape and lose it.

Your submitted program must recover the word from a snow-pit table it has never seen, so it has to derive the answer from the fitted tree rather than have it written in.

## Flag
SIGMOID_<WORD>, where <WORD> is a single uppercase word recovered from the fitted tree's structure. The practice dataset resolves to SIGMOID_DEMO. That is the worked example, not the flag; it exists so you can confirm your method works before submitting.

## Task input

| File | Rows | Purpose |
|---|---|---|
| train-input.csv | 6,000 | Practice table. Develop and test your method here |
| train-output.txt | 1 line | The answer for the practice table, so you can verify |

A third table, test-input.csv, is used at evaluation time.

Environment: Python 3.12 with scikit-learn==1.7.2, pandas==2.3.3 and numpy==2.2.6 (pinned in requirements.txt).

## Expected output
Submit a single file, main.py. When run, it must:
- read test-input.csv from its own working directory, by bare filename, with no path prefix and no directory assumptions;
- write test-output.txt into that same directory.

test-output.txt must contain the flag and nothing else: one line, uppercase, no quotes, no surrounding text. A trailing newline is fine.

## Scoring
- test-output.txt matches the expected flag: 100 points (PASS).
- Anything else, including a wrong word, missing file, crash or timeout: 0 points (FAIL).
