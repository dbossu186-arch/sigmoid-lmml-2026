# MOLDOVA: "Feather Stories" (Hard)

**Flag: `SIGMOID_<redacted>`** (accepted on the platform, 300/300)

The stork's ring is **ANSEMO**. The model narrows the answer down to two rings, both of which undo the systematic misreading at the stork's site (Lake Beleu): **XWBPHD** and **ANSEMO**. The model can't decide between them from the stork's own reads. It ranked XWBPHD first (about 64 %) from how common each site type is elsewhere, but the platform rejected XWBPHD and accepted ANSEMO.

Attempts, in order:
1. `RHGLWQ` (rejected): the plain consensus, i.e. the systematic misreading.
2. `XWBPHD` (rejected): the B-flip of the consensus.
3. `ANSEMO` (accepted): the A+B-flip, which un-flips both letter classes.

See [Outcome and why ANSEMO](#outcome-and-why-ansemo) below.

Run it with `python -I solve.py data .`. The script reads `data/observations.csv` and `data/ring_register.csv`, prints the evidence and the flag, and writes `fig_sites.png`, `fig_votes.png` and `fig_map.png`. The run on the GPU box is logged in `run.txt`.

## What generates the misreads

1. **Look-alike groups.** A letter is only ever misread as one of its three look-alikes. The six groups are RAXK, MNWH, SZBG, PLEF, QDOC and JTYI. Every registered ring has a group signature that is unique within its species, so a read can be matched to its bird even when all six letters are wrong. Of 3,023 reads, 2,878 match exactly one registered bird, 121 match several because of `?`, and 24 match none.
2. **The misread is a property of the place.** Inside a group a letter is two bits (A, B), and `read = true XOR offset`. The offset is drawn independently for each letter from a distribution that belongs to the locality and to the letter class (RAXK/SZBG/QDOC behave together, MNWH/PLEF/JTYI behave together). Inside a locality the offset does not depend on anything else:

   | factor tested inside a site | site-classes | chi-square | dof | p |
   |---|---|---|---|---|
   | bird | 81 | 554 | 580 | 0.77 |
   | observer group | 54 | 737 | 702 | 0.17 |
   | year | 81 | 570 | 532 | 0.13 |
   | group within class | 94 | 395 | 354 | 0.07 |
   | position on the ring | 66 | 550 | 547 | 0.46 |
   | true letter | 57 | 671 | 665 | 0.43 |
   | **between sites** | | **13,619** | 225 | ~0 |

   So "they do not always see the same thing" is about where they stand, not who they are. Overall 58.9 % of the letters written are not the registered letter, which is why a Hamming-distance match (first attempt) saw 96 phantom "unregistered birds".
3. **Two kinds of site.**
   - **Honest** (34 site-classes): the true letter is the most common one. The rest goes to one look-alike partner (up to about 45 %). The diagonal letter (both bits wrong) is written 7 times in 4,140 letters (0.17 %).
   - **Flipped** (41 site-classes): a wrong letter is the most common one, with 50–97 % of the reads, and the rest is spread over the other three letters. The registered letter gets 587 of the 1,848 background letters (32 %), so it is not favoured over the other two.

   A site whose minority letters cover all three look-alikes is therefore a site whose majority letter is wrong (`fig_sites.png`).
4. **Which flips occur.** Per site (even class, odd class): A-flip on the even groups only (2 sites), A-flip on the odd groups only (6), B-flip on all letters (6, plus one mixed), A+B-flip on all letters (5, plus one mixed), honest (10). An A-flip on all letters never occurs.

I found nothing that predicts a site's type: latitude, longitude, district, number of checklists, species or observers, first or main observer, years, counts and erasure rate all fail (random forest at baseline accuracy). Sites less than about 300 m apart usually share a type; beyond 0.5 km they do not.

![sites](fig_sites.png)

## The unregistered stork

The 24 unmatched reads are all White Stork at locality L5819951 (Cahul, 45.606°N 28.121°E, Lake Beleu): 24 checklists, 24 observers, 2016–2025. No other ring of any species is read there, and no read anywhere else is compatible with this bird, so the site cannot be calibrated with a registered bird. The nearest calibrated site is 64 km away.

Per-position votes:

| pos | group | letters written | consensus | XWBPHD | ANSEMO |
|---|---|---|---|---|---|
| 1 | RAXK | R 19, A 2, K 1, X 1 | R | X | A |
| 2 | MNWH | H 19, M 2, N 1 | H | W | N |
| 3 | SZBG | G 13, Z 3 | G | B | S |
| 4 | PLEF | L 15, F 2, P 1, E 1 | L | P | E |
| 5 | MNWH | W 18, N 2, H 1, M 1 | W | H | M |
| 6 | QDOC | Q 17, D 1, C 1 | Q | D | O |

Relative to the consensus RHGLWQ, the minority letters are spread over all three look-alikes in both classes: even 49 / 5 / 2 / 2, odd 52 / 6 / 2 / 3 (same / A / B / A+B). At an honest site the diagonal letter appears in under 0.5 % of letters; seeing 2 of 58 and 3 of 63 has probability 0.034 and 0.004 (1.4·10⁻⁴ together). The site sits on the flipped band in `fig_sites.png`. **RHGLWQ is the systematic misreading, not the ring.**

Both classes are flipped, so the flip is one that changes all six letters. Only two such flips exist in the data:

- B-flip: RHGLWQ → **XWBPHD**
- A+B-flip: RHGLWQ → **ANSEMO**

The reads themselves cannot separate these two, because the background of a flipped site does not favour the true letter. What separates them is how common each kind of site is. `solve.py` treats every calibrated site (31 with ≥ 30 letters per class) as one possible confusion for L5819951 and sums the likelihood over them for all 4,096 rings with this signature:

| ring | posterior | sites that support it |
|---|---|---|
| **XWBPHD** | **0.64** | L10196035, L16602511 |
| ANSEMO | 0.33 | L4521004, L8150848 |
| RHGLWQ | 0.02 | L13342603 |
| RMGFNQ | 0.005 | L4538017 |

Robustness: with ≥ 60 letters per class 0.66 / 0.34 / 0.00; with a weaker prior 0.63 / 0.37 / 0.00; resampling the sites 1,000 times XWBPHD wins 697, ANSEMO 293, RHGLWQ 8. With sites down to 15 letters RHGLWQ rises to 0.27, but only through one 21-letter site where smoothing adds diagonal mass.

Two further points for XWBPHD:

- The only two other sites with a single ringed bird are both B-flipped: L8271944 (22 reads on 22 checklists, 30 observers, the closest analogue of L5819951) and L9155654.
- A+B-flipped sites are large multi-species colonies with 5–8 ringed species; B-flipped sites have 1–7 (Mann–Whitney p = 0.003). L5819951 has one.

One point for ANSEMO: the A+B-flipped sites lie further south than the B-flipped ones (median 46.9°N against 47.1°N, p = 0.04), and the only flipped sites in the far south (Gagauzia) are A+B. I give this less weight because site types are not spatially coherent beyond 0.5 km.

![votes](fig_votes.png)
![map](fig_map.png)

## Candidates

| ring | what it is | verdict |
|---|---|---|
| **ANSEMO** | A+B-flip of the consensus | **Answer, accepted on the platform.** The only all-letter flip seen at the far-south (Gagauzia) flipped sites |
| XWBPHD | B-flip of the consensus | The model's first pick (0.64) because this flip is more common overall; rejected on the platform |
| RHGLWQ | plain consensus | Rejected on the platform; the even background shows the site is flipped |
| KMZFNC | A-flip of all letters | No site in the data has this flip |
| KHZLWC, RMGFNQ | flip of one class only | Would need the other class to be honest, but both classes have an even background (posterior ≤ 0.005) |
| KWZPHC and other mixed flips | A on even, B on odd | Such sites have a 50/50 split on the even letters; L5819951 has 84 % |
| RHZLWQ | consensus with pos 3 = Z | Z is the 3-of-16 minority at pos 3 |
| EKBDAO, JGPDZT, IYRNDQ, DLNSLR, WCNBYA and the other "unregistered clusters" of the first attempt | registered storks FAGQKC, JZPOGT, IYKNOC, DFWSFR, HOMBIK … read at flipped sites | Decoys. Each has the signature of a registered stork ringed at that site |

## Outcome and why ANSEMO

The model leaves two rings, XWBPHD and ANSEMO. Both un-flip every letter of the Lake Beleu consensus, but they assume different site types: B-flip for XWBPHD, A+B-flip for ANSEMO. The stork's own reads can't separate them, because at a flipped site the letters that are not the consensus don't point to the true one. The ranking above therefore relies on other sites.

- **Model ranking:** the B-flip is the more common site type overall, and it is the type at the two other single-bird sites. That made XWBPHD the first pick. It was rejected.
- **The geographic clue:** the only flipped sites in the far south, in Gagauzia, the region nearest the Lower Prut, are A+B-flipped. A+B-flipped sites as a group also lie further south (p = 0.04). I gave this clue less weight because a site's type does not carry over to sites more than 0.5 km away. The platform result says it was the right clue to follow: the author evidently gave Lake Beleu the flip type of the far-south sites.
- **Accepted:** `SIGMOID_<redacted>`, 300/300.

Lesson: when a site can't be calibrated, its closest neighbours (here the far-south sites) were a better guide than how common each site type is across the whole dataset.

## First attempt (rejected): `SIGMOID_<redacted>`

The first solver matched reads to the register by Hamming distance, estimated one global error rate (11.7 %) with per-observer weights, clustered the leftovers and took the consensus of the Lower Prut cluster. It picked the right bird and the right place. It was wrong on two assumptions:

- It treated misreads as random noise around the truth. They are systematic per site, and at flipped sites the majority letter is wrong.
- It learned the error model only from reads within one letter of a registered ring, that is only from honest sites, so it never saw the flips.

Leave-one-observer-out and the bootstrap all returned RHGLWQ because every observer at the site makes the same systematic misreading.

## Limits

- The choice between XWBPHD and ANSEMO can't be made from the data alone, because L5819951 can't be calibrated. The model ranked XWBPHD first; the platform accepted ANSEMO, the type of the nearest far-south flipped sites.
- The dichotomy "even background means flipped" is empirical: 34 honest and 41 flipped site-classes, no exception among those with ≥ 30 letters.

## Files

- `solve.py`: end-to-end solver, reads `data/`, prints the flag.
- `run.txt`: log of the run on the GPU box.
- `fig_sites.png`: honest and flipped sites, with L5819951 marked.
- `fig_votes.png`: letters written at L5819951 with the consensus and the two candidate rings.
- `fig_map.png`: calibrated sites and L5819951.
- `NOTES_after_rejection.md`: working notes written after the rejection (superseded by this file).
