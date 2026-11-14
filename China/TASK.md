# CHINA - Carved in bones, engraved in the heart (Medium, CV)

## Storyline (summary)
Rubbings copy carved surfaces (carved lines stay white). Oracle bones from Yinxu (Anyang), pit YH127 (1936, 17,000+ pieces). An Anyang archivist kept rubbings in eight crates; each crate had one glyph painted on its lid and held five bundles (sets) of rubbings — five, as diviners often put one question to five shells. Saying: "The bones answer whoever counts carefully enough."

## Objective
Each crate keeps one letter. It is written in how often the crate's glyph was carved across its five sets. Gather the letters and form the word.

## Flag
SIGMOID_<WORD>: one word of eight letters, uppercase.

## Input
Greyscale JPEG rubbings of inscribed bones and shells.
- train/images/: 1,800 labelled rubbings.
- train_labels.csv: one row per character: image, x, y, w, h, glyph_id (pixel boxes from top-left). Uncommon glyphs share id `rare`.
- glyphs.csv: glyph_id, modern_character, train_boxes.
- crates.csv: crate, glyph_id, modern_character — the glyph of each crate.
- crates/crate_XX/set_Y/: 1,200 unlabelled rubbings, 30 per set (8 crates x 5 sets). File names carry no meaning.
- baseline_solution.py shows input and output formats.

## Expected output (submit on platform)
- The flag SIGMOID_<WORD>.
- A URL with code proving the solution, a counts.csv (crate, set, glyph_id, count; one row per set) and a short write-up.

## Scoring
Correct flag with supporting code, counts and write-up: 200 pts (PASS). Incorrect flag, or correct flag without supporting material: 0.
Hints 1-3 available (penalty) — do not unlock.

## Data
https://drive.google.com/drive/folders/1TE_-nwZzBUTDztXivSbNqeyCFFaN1WDa
