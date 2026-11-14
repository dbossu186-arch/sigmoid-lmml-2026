# Portugal — The Douro Paradox (Medium, Classic ML & Data Science)

## Storyline
The Douro Valley's viticulture institute has trialled a new drip-irrigation method against the old one. The results look unambiguous: plots on the new method succeed noticeably more often. Before the institute recommends it valley-wide, a skeptical intern insists on splitting the data — and finds that, split the right way, the new method is worse than the old one in every single group.

Your job is to find the variable that undoes the "win".

## Objective
Overall, treatment has a higher harvest_success rate than control. Find the single column that, when the data is split by it into the right groups, makes treatment worse than control within every group — a complete Simpson's Paradox reversal. Several other columns look plausible but do not produce a full reversal.

## Flag
SIGMOID_<PHRASE>, where <PHRASE> is the name of the identified column in UPPERCASE, underscores kept (e.g. soil_ph → SIGMOID_SOIL_PH).

## Input
douro_vineyard_data.csv (data/) — 3,543 rows, one per experimental plot:

| Column | Description |
|---|---|
| plot_id | Unique plot identifier |
| treatment | control (old method) or treatment (new drip irrigation) |
| soil_depth_cm | Soil depth in cm |
| trellis_system | vertical shoot positioning, lyre, bush vine |
| vineyard_age_group | young (<15y), mature (15-40y), old (>40y) |
| sun_hours_july | Sunshine hours in July |
| vine_density_per_ha | Vines per hectare |
| cooperative | Winemaking cooperative the plot belongs to (COOP-01 … COOP-12) |
| annual_rainfall_mm | Annual rainfall in mm |
| grape_variety | Touriga Nacional, Touriga Franca, Tinta Roriz |
| distance_from_porto_km | Road distance from Porto in km |
| organic_certified | yes / no |
| elevation_m | Elevation in meters |
| soil_ph | Soil pH |
| harvest_month | September or October |
| slope_aspect | north-facing / south-facing / east-facing / west-facing |
| harvest_success | 1 if the plot met the yield/quality target, else 0 |

## Expected output
- The flag, SIGMOID_<PHRASE>.
- A URL (GitHub, Google Drive, Colab, ...) to a notebook or write-up showing:
  - the overall treatment vs control success rates (treatment ahead);
  - how you split the identified column into groups, and the success rates within every group (control ahead in each);
  - one sentence on why pooling the groups produces the reversal.

## Scoring
Flag Evaluation; organizers review the linked write-up manually.
- Correct flag and a write-up showing the reversal in every group: 200 points (PASS)
- Incorrect flag, or a correct flag without a supporting write-up: 0 points (FAIL)

Hints 1–3 exist (locked; unlocking carries a penalty — do not unlock).

## Data link
https://drive.google.com/drive/folders/1NEtS0uwNpa_1iI0nZkXToIvLrk7CaKoD
