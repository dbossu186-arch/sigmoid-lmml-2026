# Finland — Pipes, Leaks, and the Graph Beneath (Medium, Classic ML & Data Science)

## Storyline
For two years, ten regional offices of the Finnish Maritime Ferry Authority have logged every inter-town route by hand, each with its own habits. The Authority now wants to know which town is the network's busiest hub, so it can prioritize it for investment. A junior analyst runs a quick group-by on the raw log and confidently reports Orivesi. Nobody checks whether the log really says what the group-by says it does.

## Objective
Clean the route log so that every town is counted as exactly one town and every connection once, rebuild the real route network, and rank its towns by number of distinct connections.

## Flag
SIGMOID_<WORD>: <WORD> is made of the first letters of the 7 busiest towns in your cleaned network, from most to fewest connections, in uppercase.

## Input (data/)
routes.csv: the raw route log, one row per log entry.

| Column | Description |
|---|---|
| log_id | Log entry ID |
| logged_on | Date the entry was logged |
| office | Office that logged the entry |
| from_town, to_town | The two ends of the route, as logged |
| route_type | ferry, train, or bus |

## Expected output
The flag, entered on the platform.

Self-check: a correctly cleaned network has 116 towns and 448 connections, and no two of the 8 busiest towns have the same number of connections.

## Scoring
- Correct flag: 200 points (PASS)
- Incorrect flag: 0 points (FAIL)

Hints 1–3 exist (locked; unlocking carries a penalty — do not unlock).

## Data link
https://drive.google.com/drive/folders/1vmtpg82kdxeFPGqs9IqUp80enMTYUTOR
