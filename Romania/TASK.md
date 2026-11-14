# ROMANIA — "Shaken, Still Standing" (Hard, Europe region)

Source: https://sigmoid.academy/tasks (task popup), collected 2026-11-14.

## Storyline (verbatim)

Imagine this. Dark, silence, peace...

And yet again... your night shift at Măgurele. On your screens, more than 311 sensors from all over Romania draw their quiet lines. Most nights nothing happens. Then one line jumps.

Deep under Vrancea, 80 to 160 km down, the ground moves every few decades, and a whole region feels it; people call the place Europe's "Ring of Fire". In 1977 it reached magnitude 7.5. Brașov, right across the mountains, got away with intensity V. The infamous president of Romania from back then was in Nigeria and missed it.

Since 2013 the early-warning system buys the country a few seconds, sometimes tens of seconds, before the strong shaking arrives. Five seconds after the first waves, the alarm has to go out. Or not. Trucks, quarry blasts and the neighbours' renovation also make lines jump.

Three networks send you their records: Moldavia, Muntenia, Transylvania. Your call: deep Vrancea again, or not?

## Objective (verbatim)

Give every record in sealed.csv a verdict (Vrancea event or not), so that you can tell, almost with certainty, when a call is an earthquake.

## Flag

Format `SIGMOID_<WORD>`.

## Task input

| File | Purpose |
|---|---|
| dataset/train.csv | 7,500 labelled records: record_id, channels c01..c48, vrancea (1 = Vrancea event) |
| dataset/sealed.csv | 5,000 records of the same kind, without vrancea |
| dataset/vault.npz | what the evaluator reads; it holds no label in the clear |
| evaluation_script.py | `python evaluation_script.py predictions.csv`, where predictions.csv has record_id,vrancea for every sealed record |
| baseline_solution.py | loads the data, fits the usual single models, writes predictions.csv, calls the evaluator |

The 48 channels are spectral amplitudes of the record.

Data link: https://drive.google.com/drive/folders/1dCTKtIKpRqyJs57fN6nqIR2RuYuwt-Hd (folder ID `1dCTKtIKpRqyJs57fN6nqIR2RuYuwt-Hd`)

## Pinned versions / limits

None stated on the task page (no Python version, package pins, timeouts or file size limits listed). Check evaluation_script.py / baseline_solution.py for any implied ones.

## Expected output

Submit the flag on the platform together with a link to your materials, explaining how you reached the result: the models you tried, what each scored, and how you combined them.

## Scoring

- Correct flag, and the linked work shows how you reached it: 300 points (PASS)
- Incorrect flag, or a flag without evidence: 0 points (FAIL)

Hints 1–3 exist (locked; unlocking carries a penalty — not unlocked).

## Notes on the wording (possible clues)

- "tell, almost with certainty, when a call is an earthquake" → precision-oriented / high-confidence positives?
- "the usual single models" in the baseline → hints that an ensemble/combination is expected ("how you combined them").
- Three networks (Moldavia, Muntenia, Transylvania) → possible sub-populations / domain shift.
- "Five seconds after the first waves" → time/feature budget.
