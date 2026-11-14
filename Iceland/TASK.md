# ICELAND: What The Data? (Easy)

## Storyline
The Icelandic Meteorological Office runs a network of geothermal spring monitoring stations across Iceland, from Geysir to Mývatn. The stations radio temperature readings back over a satellite link that's been in service since long before anyone on the current team was hired. It works, mostly. Retransmissions duplicate readings, the export format is a relic from whenever someone last touched the legacy system, station names with Icelandic letters arrive garbled, and firmware updates never quite made it to every station.

You've been handed one export to practise on. Write a cleaner that fixes it, because the season's real export is waiting, and it's messier than the one you've seen.

## Objective
Write a Python script, main.py, that turns a damaged station export into a clean table in the exact target format given under Expected output.

The damage comes in three layers, and every export contains all of them:
1. **Duplicates and number formatting.** The satellite link retransmits, so some readings appear more than once. Some temperatures use a comma as the decimal separator (54,6), a European locale habit, which is why the file uses `;` as the column delimiter.
2. **File structure and encoding.** A legacy preamble of comment lines (starting with #) and blank lines sits before the real header, and its length varies between exports. Some station names arrive mis-decoded: a genuine encoding mismatch, not a typo.
3. **Units.** A unit column says whether each reading is in Celsius (C) or Fahrenheit (F). Some stations never got the firmware update and still report in Fahrenheit.

Your script will be run on an export it has never seen. It is messier than the practice file (more duplicates, more broken names, more Fahrenheit readings, a longer preamble) and includes stations that do not appear in the practice file. Fix the problems generically rather than hardcoding values.

## Flag
Format: SIGMOID_<WORD>. The platform releases the flag once your main.py produces the exact expected output on the held-back export.

## Task input

| File | Rows | Purpose |
|---|---|---|
| train-input.csv | 30 readings plus duplicates | Practice export. Develop and test your cleaner here |
| train-output.csv | 30 | The correct cleaned version of the practice export, so you can verify your output |

At evaluation time your script receives a held-back export named test-input.csv. It has the same raw schema as train-input.csv but different, messier data.

Raw input format (`;`-delimited, UTF-8), after the preamble:
`reading_id;station_name;temperature_raw;unit;reading_date`

## Expected output
Submit a single file, main.py. When run, it must:
- read test-input.csv from its own working directory, by bare filename, with no path prefix;
- write test-output.csv into that same directory, in the target clean format below.

Target clean format (`,`-delimited, UTF-8): `reading_id,station_name,temperature_c,reading_date`
- reading_id: integer, exactly one row per reading, rows sorted by reading_id ascending
- station_name: the correct station name, with its Icelandic characters restored
- temperature_c: Celsius, written with exactly one decimal place (e.g. 85.0, 61.3)
- reading_date: YYYY-MM-DD, unchanged

No index column, no extra columns, no quoting.

To test locally, copy train-input.csv to test-input.csv, run `python main.py`, and compare the test-output.csv it writes with train-output.csv. They must be identical.

Runtime: Python 3.10–3.12 with pandas==2.2.2 and numpy==1.26.4 available (the standard library alone is enough). No internet access, non-interactive, CPU only, 60-second time limit.

## Scoring
Your test-output.csv is compared with the expected cleaned table, after normalising line endings and trailing whitespace at the end of the file. Every other character must match.
- Output matches the expected table exactly: 100 points (PASS), flag released.
- Any difference (a missed duplicate, one broken station name, a wrong conversion, a missing decimal place), or a script that crashes, times out or writes no test-output.csv: 0 points (FAIL).
