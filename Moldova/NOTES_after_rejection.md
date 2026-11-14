# Moldova: notes after SIGMOID_<redacted> was rejected (2026-11-14)

## What holds
- The letters fall into 6 closed look-alike groups, and misreads (about 99% of them) stay inside a group:
  RAXK, MNWH, SZBG, PLEF, QDOC, JTYI (U and V are never used).
- Each registered ring has a group signature (the group of each letter) that is unique within its species. Under that matching, 2,999 of 3,023 reads fit a registered bird.
  Exactly one unregistered bird is left: the 24 White Stork reads at L5819951 (Cahul, Lake Beleu, 45.606 N, 28.121 E), with signature 012314.
  The site and the bird are therefore right. The open question is which letter inside each group.

## Why the plain consensus may be wrong
- At many localities, registered rings are read with the SAME consistent wrong letters across different observers and years.
  Examples: ADHIOL is read as XOHIDL at L5562838; SBSSPQ as BSBBLC at L4597728; FAGQKC as EKBDAO.
- The distortion depends on the locality, not the observer: different observers at the same locality produce the same mapping.
- Take each group's letter index (0–3). The mapping is read_idx = true_idx XOR key[group], where key is a per-locality 6-vector (Klein four-group per group).
- The keys observed are combinations of three generators:
  K1=302030, K2=030202, K3=211111, and K1^K2^K3=123323 (seen at L8150848, L4521004, L4874209).
- 90–100% of reads within a locality share one key (the exception is L7513445, which is mixed).
- None of these predicts the key: latitude/longitude, district, observer, year, month, hour, or species present.
- L5819951 has no registered bird to calibrate its key, so its key is unknown. Identity gives RHGLWQ (rejected).

## Candidate words (RHGLWQ with each key applied)
| key | word |
|---|---|
| 0 (identity) | RHGLWQ (rejected) |
| K1 | KHZLWC |
| K2 | RMGFNQ |
| K3 | XWBPHD |
| K1^K2 | KMZFNC |
| K1^K3 | AWSPHO |
| K2^K3 | XNBEMD |
| K1^K2^K3 | ANSEMO |
