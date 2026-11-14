# MOLDOVA — "Feather Stories" (Hard, Europe region)

Source: https://sigmoid.academy/tasks (task popup), collected 2026-11-14.

## Storyline (verbatim)

In the far south of Moldova, the Prut slows down before it meets the Danube. Lakes Beleu and Manta spread across the floodplain. In 2018 UNESCO added this corner of the country to its Man and the Biosphere Programme as the Lower Prut Biosphere Reserve, the first in Moldova. The plan is to join it one day with the Danube Delta across the border, in one reserve shared with Romania and Ukraine.

The reserve belongs to the birds. Pelicans rest on the lakes on their way to the Delta. Herons and egrets stand in the shallows. Each spring the white storks return from Africa, most of them to the nest they left. Moldovans know this bird well: a stork carrying grapes is a national symbol, and it marks the country's wine.

Some of these birds wear a ring on the leg. It is a small thing with a short code, but it turns a bird into an individual with a history. Birdwatchers read it through a telescope, from far away, and write down what they see. They do not always see the same thing.

One stork of the Lower Prut has been read many times. Nobody has written it into the register, though.

## Objective (verbatim)

In this dataset every ring carries six letters. A birdwatcher who spots one writes down what they could read. A letter they could not make out is written as `?`. A letter they did make out is not always the right one.

The ringing register lists every ringed bird in the data except one White Stork. Find what is written on its ring.

## Flag

`SIGMOID_<WORD>`: `<WORD>` is the six uppercase letters on the stork's ring.

## Task input

Two CSV files.

**observations.csv**: bird records from Moldova, one row per species on a checklist (79,968 rows).

| Column | Meaning |
|---|---|
| record_id | Record identifier |
| checklist_id | Checklist the record belongs to: one outing, one place, one start time |
| date, time | Start of the checklist, local time |
| latitude, longitude | Where the checklist was made |
| locality_id | Identifier of that place |
| district | District of Moldova |
| observers | Observer identifiers, separated by `\|` when several people shared the checklist |
| species, common_name | Scientific and English name |
| count | Number of individuals, when it was noted |
| ring_read | The ring as it was read on that sighting; empty when no ring was noted |

**ring_register.csv**: the ringed birds (289 rows).

| Column | Meaning |
|---|---|
| ring | The true inscription |
| species | Scientific name |
| ringed_latitude, ringed_longitude | Where the bird lives |
| ringed_year | Year it was ringed |

Data link: https://drive.google.com/drive/folders/1R_cYNcO-t0MRR2ri6Nhgj1cSJ7zCiWBM (folder ID `1R_cYNcO-t0MRR2ri6Nhgj1cSJ7zCiWBM`)

## Pinned versions / limits

None stated (flag task, no code evaluator).

## Expected output

The flag, submitted on the platform, together with a link to your code and a short explanation of how you found it.

## Scoring

- Correct flag, with relevant explanations: 300 points (PASS)
- Incorrect flag or lack of explanation: 0 points (FAIL)

Hints 1–4 exist (locked; penalties — not unlocked).

## Wording clues

- "One stork of the Lower Prut" → the unregistered White Stork lives near Lakes Beleu/Manta (Cahul district, far south), not elsewhere.
- "most of them to the nest they left" → site fidelity: same locality across years.
- "read many times" → consensus over many noisy reads (`?` = erasure, other letters = substitutions; observer-specific error rates likely).
- "They do not always see the same thing" → observers differ in reliability; possible decoy unregistered storks elsewhere or other species.
