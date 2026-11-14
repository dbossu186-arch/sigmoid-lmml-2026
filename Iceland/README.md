# Iceland: What The Data? (Sigmoid LMML 2026, Easy)

Cleans a geothermal station export. It reads `test-input.csv` and writes `test-output.csv` with the columns `reading_id, station_name, temperature_c, reading_date`. Standard library only.

What it fixes, generically:
- a preamble of `#` comment lines and blank lines before the real header, of any length;
- duplicated readings (satellite retransmissions), keeping one row per `reading_id`;
- decimal commas, stray whitespace and degree signs in temperatures;
- Fahrenheit readings, converted to Celsius with one decimal place;
- mis-decoded station names: UTF-8 bytes read as Latin-1 / Windows-1252, even twice.
