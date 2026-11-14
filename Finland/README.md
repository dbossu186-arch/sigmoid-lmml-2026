# Finland — Pipes, Leaks, and the Graph Beneath

**Flag: `SIGMOID_<redacted>`**

Reproduce with `python -I solve.py`. It prints the counts, the top 10 and the flag, and writes `top15_degree.png`.

## Result
Cleaned network: **116 towns, 448 connections**. The top 8 all have different degrees, so the self-check passes.

| # | Town | Degree |
|---|------|--------|
| 1 | Helsinki | 39 |
| 2 | Orivesi | 38 |
| 3 | Tornio | 33 |
| 4 | Lieksa | 29 |
| 5 | Imatra | 27 |
| 6 | Nokia | 23 |
| 7 | Eura | 22 |
| 8 | Närpiö | 21 |
| 9 | Kemiönsaari | 19 |
| 10 | Inari / Kuopio | 14 |

H-O-T-L-I-N-E gives `SIGMOID_<redacted>`.

The trap: counting raw strings ranks Orivesi first (44 rows). Helsinki looks smaller because about a third of its routes are logged as *Helsingfors*. Raw rows also count repeated entries.

## Data profile
- 838 rows. Every row has 6 fields, with no shifted or merged columns. Dates fall between 2024-01-01 and 2025-12-29, all valid. Offices are OFF-01..OFF-10. `route_type` is always one of ferry, bus or train.
- Every office has the same surface noise in about the same amounts: ~15–28% leading or trailing spaces, ~5–14% a trailing `.`, ~7–15% ALL CAPS, ~7–14% all lowercase, and ä/ö/å sometimes written as a/o/a.
- **Office-specific habit:** OFF-03, OFF-07 and OFF-09 write about **38–39%** of town names in **Swedish** (Helsingfors, Torneå, Närpes, Åbo, ...). The other offices do this only 4–9% of the time.
- No self-loops remain after cleaning. 390 rows repeat a pair that is already logged, and 151 pairs are logged in both directions.

## Cleaning rules
1. **Trim whitespace** at both ends. The spaces are typing noise.
2. **Drop a trailing period** (`Turku.`). It is a typing habit, not part of the name.
3. **Lower-case**. Case does not change which town is meant.
4. **Fold diacritics** (ä→a, ö→o, å→a) with Unicode NFKD, then remove the combining marks. Some offices type without diacritics (`Jyvaskyla`, `Pieksamaki`, `Mantsala`).
5. **Map Swedish or alternative names to the Finnish name**. There are 61 aliases, e.g. Helsingfors→Helsinki, Torneå/Tornea→Tornio, Närpes→Närpiö, Kimitoön→Kemiönsaari, Åbo→Turku, Björneborg→Pori, Villmanstrand→Lappeenranta, Ingå→Inkoo, Ilmola→Ilmajoki, Träskända→Järvenpää, Lundo→Lieto, Vörå→Vöyri, plus `Keuru`→Keuruu (a truncated spelling). The full table is `ALIAS` in `solve.py`.
6. **Undirected edges.** A→B and B→A are the same connection, so each pair is stored as a sorted tuple.
7. **Deduplicate.** A connection counts once, however many times or by however many offices it was logged.
8. **Drop self-loops** after canonicalisation. None turned up, but the rule is in place.
9. **Ignore `route_type`** for edge identity. A connection is a pair of towns. In this data it makes no difference: counting (pair, type) also gives 448, because each pair appears with only one type.
10. **Keep every row.** There are no out-of-range dates, malformed rows or test rows to remove.

Degree is the number of distinct neighbours. Ties are broken alphabetically, which only affects places 10 and below.

## Why the self-check pins the answer
- Leave-one-out test on the alias table: removing **any single alias** gives 117 towns. Every alias is needed for 116, and none is spare.
- Merging any two of the 116 canonical names would give fewer than 116 towns. Dropping any non-duplicate row would give fewer than 448 connections. The rules hit 116/448 exactly, with nothing left over.
- The top 7 have degrees 39 > 38 > 33 > 29 > 27 > 23 > 22, so there are no ties. The only close gap is Helsinki vs Orivesi (39 vs 38). Helsinki leads only once Helsingfors is merged, which is the intended trap. Without that merge the count would be 117 towns / 460 connections, which fails the check.
