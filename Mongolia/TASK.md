# MONGOLIA: The Baa-rilliant task! (Medium, CV)

Source: https://sigmoid.academy/tasks (popup text copied 2026-11-14)

## Storyline
On the Mongolian steppe a family's savings account has four legs and eats grass. Herding families keep sheep, goats, cattle, horses and camels, move camp with the seasons, and the herder spends the day on horseback (or, lately, a motorbike) riding circles around the flock. Sheep are by far the most numerous: at the end of 2023 the country counted 64.7 million head of livestock, and the brutal winter that followed killed more than 7 million of them. By the end of 2024 the count stood at 57.6 million, 24.5 million of them sheep.

So every herder counts, and every herder knows how badly it goes. Sheep refuse to stand in a line. They bunch into one woolly blob the moment they notice you, the far ones hide behind the near ones, the dark ones vanish in the shade of a bush, and the whole flock moves while you are at "forty-seven". Count them twice and you get two numbers; count them three times and you get a third.

These days a drone goes up, hangs over the pasture and photographs the flock all at once. The sheep still stand in a blob. They just do it from above now.

## Objective
Count the sheep in every photo of the drone survey and hand the census office what it asked for: a count for each surveyed plot. The office knows where every photo was taken; what it does not know is how many sheep were in it. **Once every plot has its number, the census tells you more than a total.**

## Flag
`SIGMOID_<WORD>` where `<WORD>` is the word the census reveals, in uppercase.

## Task input
- train/: 280 photos, train_000.jpg … train_279.jpg.
- train_counts.csv: image,count (exact number of sheep in each training photo).
- train_points.csv: image,x,y (one dot per sheep, centre of its body).
- survey/: the 333 survey photos (test data), survey_0001.jpg … survey_0333.jpg, one per plot.
- survey_plots.csv: the drone's flight log: image,x_m,y_m, the position of each photographed plot in local survey coordinates (metres).

Training and survey photos come from the same flights over the same pastures, at the same range of altitudes and view angles. The only difference is that nobody counted the survey photos.

## Expected output
Submit on the platform:
1. The flag `SIGMOID_<WORD>`.
2. A URL with your code and your counts.csv for the survey photos (image,count, one row per photo), proving how you found the flag.

## Scoring
- Correct flag, with counts.csv and solution proof: 200 points (PASS)
- Incorrect flag, or a correct flag without supporting counts and write-up: 0 points (FAIL)

## Submission
Flag field + link. Four hints (locked; don't unlock without asking).

## Data
Google Drive folder ID `1acU-wiKH9ar7uTmg3m8-v2aAtEvT4k9e` → `data/`
