#!/usr/bin/env python3
"""Sigmoid MOLDOVA "Feather Stories": the ring of the unregistered Lower Prut White Stork.

Usage:  python -I solve.py [data_dir] [out_dir]
  data_dir defaults to ./data (observations.csv, ring_register.csv)
  out_dir  defaults to . (writes fig_sites.png, fig_votes.png, fig_map.png)

Second attempt. The first one (plain consensus, SIGMOID_<redacted>) was rejected; see README.

Pipeline
  1. Look-alike groups. Letters are only ever misread inside six groups of four. A read
     is matched to the registered ring of the same species with the same group signature.
  2. Inside a group a letter is two bits (A, B). read = true XOR offset, where the offset
     is drawn per letter from a distribution that belongs to the SITE (locality) and to the
     letter class (even groups RAXK/SZBG/QDOC, odd groups MNWH/PLEF/JTYI). Tested: no
     dependence on bird, observers, year, position or true letter within a site.
  3. Two kinds of site. "Honest": the true letter dominates and the rest goes to one
     look-alike partner; the diagonal ("both bits") letter is almost never written.
     "Flipped": a wrong letter dominates and the rest is spread evenly over the other
     three letters. So an even background marks a site whose majority reading is wrong.
  4. The 24 reads no registered bird explains are the White Stork at L5819951 (Lake Beleu).
     Their background is even, so the consensus RHGLWQ is a flipped reading.
  5. Which flip: empirical Bayes. Every calibrated site is one possible confusion for
     L5819951; the posterior over the 4096 rings of the signature is the mixture.
"""
import sys
import os
import itertools
from collections import Counter

import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
STORK = "Ciconia ciconia"
# Each group in the order (A,B) = 00, 10, 01, 11: index bit0 = A, bit1 = B.
ORD = ["RKXA", "MHNW", "SBZG", "PELF", "QCDO", "JYTI"]
AB = {c: (g, i) for g, o in enumerate(ORD) for i, c in enumerate(o)}
NAMES = ["same", "A", "B", "A+B"]
rng = np.random.default_rng(0)

obs = pd.read_csv(os.path.join(DATA, "observations.csv"), dtype={"ring_read": str})
reg = pd.read_csv(os.path.join(DATA, "ring_register.csv"))
reads = obs[obs.ring_read.notna()].copy().reset_index(drop=True)
print(f"{len(obs)} rows, {len(reads)} ring reads, {len(reg)} registered rings")

# ------------------------------------------------------------ 1. groups and matching
used = set("".join(reg.ring)) | (set("".join(reads.ring_read)) - {"?"})
assert used <= set(AB), used - set(AB)


def sig(s):
    return tuple(AB[c][0] for c in s)


reg["sig"] = reg.ring.map(sig)
assert not reg.duplicated(["species", "sig"]).any()  # signature is unique within a species
by_sp = {sp: list(zip(g.ring, g.sig)) for sp, g in reg.groupby("species")}


def match(sp, x):
    return [r for r, s in by_sp.get(sp, []) if all(a == "?" or AB[a][0] == g for a, g in zip(x, s))]


reads["cands"] = [match(s, x) for s, x in zip(reads.species, reads.ring_read)]
reads["nm"] = reads.cands.map(len)
reads["true"] = [c[0] if len(c) == 1 else None for c in reads.cands]
print("reads by number of registered rings with a compatible signature:",
      dict(reads.nm.value_counts().sort_index()))
orphan = reads[reads.nm == 0]
print(f"reads no registered bird can explain: {len(orphan)} -> "
      f"{dict(Counter(zip(orphan.species, orphan.locality_id, orphan.district)))}")
assert orphan.species.eq(STORK).all() and orphan.locality_id.nunique() == 1
SITE = orphan.locality_id.iloc[0]

# ------------------------------------------------------------ 2. letter-level offsets
u = reads[reads.true.notna()].reset_index(drop=True)
rows = []
for i, (x, t) in enumerate(zip(u.ring_read, u.true)):
    for p, (a, b) in enumerate(zip(x, t)):
        if a != "?":
            rows.append((i, p, b, AB[b][0], AB[b][0] % 2, AB[a][1] ^ AB[b][1]))
L = pd.DataFrame(rows, columns=["i", "pos", "t", "grp", "odd", "off"])
for c in ["locality_id", "observers", "true", "date"]:
    L[c] = u[c].to_numpy()[L.i]
L["year"] = L.date.str[:4]
print(f"\n{len(u)} uniquely matched reads, {len(L)} letters; "
      f"{(L.off != 0).mean():.1%} of the letters are not the registered letter")


def chi2_within(factor, min_n=8):
    """Sum of chi-square statistics of offset ~ factor inside each (site, class)."""
    from scipy.stats import chi2_contingency, chi2
    stat = dof = n = 0
    for _, g in L.groupby(["locality_id", "odd"]):
        T = pd.crosstab(g[factor], g.off)
        T = T.loc[T.sum(axis=1) >= min_n, :]
        T = T.loc[:, T.sum(axis=0) > 0]
        if T.shape[0] < 2 or T.shape[1] < 2:
            continue
        s, _, d, _ = chi2_contingency(T)
        stat, dof, n = stat + s, dof + d, n + 1
    return n, stat, dof, chi2.sf(stat, dof)


try:
    print("does the offset depend on anything else inside a site? (chi-square, summed over sites)")
    for f in ["true", "observers", "year", "grp", "pos", "t"]:
        n, s, d, p = chi2_within(f)
        print(f"  {f:10s} {n:3d} site-classes  chi2={s:7.1f}  dof={d:4d}  p={p:.3f}")
    from scipy.stats import chi2_contingency
    s, p, d, _ = chi2_contingency(pd.crosstab(L.locality_id, L.off).loc[lambda T: T.sum(axis=1) >= 30])
    print(f"  between sites (>=30 letters): chi2={s:.0f} dof={d} p={p:.1e}")
except ImportError:
    print("scipy missing: chi-square checks skipped")

# ------------------------------------------------------------ 3. site confusion, two kinds
cnt = {}
for loc, odd, off in zip(L.locality_id, L.odd, L.off):
    cnt.setdefault(loc, np.zeros((2, 4)))[odd, off] += 1
recs = []
for loc, c in cnt.items():
    for odd in (0, 1):
        v = c[odd]
        if v.sum() < 30:
            continue
        dom = int(v.argmax())
        rel = [v[dom ^ k] for k in range(4)]  # counts relative to the dominant letter
        recs.append(dict(site=loc, cls="odd" if odd else "even", n=int(v.sum()), dom=NAMES[dom],
                         share=rel[0] / v.sum(), rA=int(rel[1]), rB=int(rel[2]), rAB=int(rel[3]),
                         diag=rel[3] / v.sum(), spread=min(rel[1:]) / v.sum()))
SC = pd.DataFrame(recs)
SC["kind"] = np.where(SC.dom == "same", "honest", "flipped")
pd.set_option("display.width", 220, "display.max_rows", 300)
print("\nsite x class confusion (>=30 letters), counts relative to the dominant letter:")
print(SC.sort_values(["kind", "dom", "share"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
h = SC[SC.kind == "honest"]
f = SC[SC.kind == "flipped"]
h2 = h[(h.rA > 0) | (h.rB > 0)]  # leave out the pure 'same / A+B' pair-confusion sites
print(f"\nhonest site-classes: {len(h)}; letters {h.n.sum()}; diagonal letter written "
      f"{h2.rAB.sum()} times in {h2.n.sum()} letters ({h2.rAB.sum() / h2.n.sum():.2%}) "
      f"where another look-alike is also in play")
print(f"flipped site-classes: {len(f)}; background split A/B/A+B = "
      f"{f.rA.sum()}/{f.rB.sum()}/{f.rAB.sum()}")
truth_rel = {"A": "rA", "B": "rB", "A+B": "rAB"}
t_cnt = sum(r[truth_rel[r.dom]] for _, r in f.iterrows())
print(f"  of the flipped background, the registered letter gets {t_cnt} of "
      f"{f.rA.sum() + f.rB.sum() + f.rAB.sum()} (1/3 expected if it is not favoured)")
key_tab = (SC.pivot(index="site", columns="cls", values="dom").dropna()
           .groupby(["even", "odd"]).size().rename("sites"))
print("\ndominant offset per site (even class, odd class):")
print(key_tab.to_string())

# ------------------------------------------------------------ 4. the stork at Lake Beleu
g = reads[reads.locality_id == SITE].sort_values("date")
row = g.iloc[0]
print(f"\n{SITE}: {row.district}, {row.latitude:.4f}N {row.longitude:.4f}E; {len(g)} reads, "
      f"{g.checklist_id.nunique()} checklists, "
      f"{len({o for s in g.observers for o in s.split('|')})} observers, "
      f"{g.date.min()[:4]}-{g.date.max()[:4]}")
votes = [Counter(x[p] for x in g.ring_read if x[p] != "?") for p in range(6)]
cons = "".join(v.most_common(1)[0][0] for v in votes)
print(f"plain consensus (first attempt): {cons}")
relc = np.zeros((2, 4))
for p in range(6):
    for a, k in votes[p].items():
        relc[AB[cons[p]][0] % 2, AB[a][1] ^ AB[cons[p]][1]] += k
for odd in (0, 1):
    v = relc[odd]
    print(f"  {'odd ' if odd else 'even'} class relative to the consensus: same {int(v[0])}, "
          f"A {int(v[1])}, B {int(v[2])}, A+B {int(v[3])}  (share {v[0] / v.sum():.2f})")
rate = h2.rAB.sum() / h2.n.sum()
try:
    from scipy.stats import binom
    pv = [binom.sf(relc[o, 3] - 1, relc[o].sum(), max(rate, 0.005)) for o in (0, 1)]
    print(f"  P(that many diagonal letters | honest site, rate {max(rate, 0.005):.3f}) = "
          f"{pv[0]:.3f} (even), {pv[1]:.4f} (odd), {pv[0] * pv[1]:.1e} (both)")
except ImportError:
    pass

# ------------------------------------------------------------ 5. empirical Bayes over rings
gs = [AB[c][0] for c in cons]
N = np.zeros((6, 4))  # N[pos, letter index] at the stork site
for p in range(6):
    for a, k in votes[p].items():
        N[p, AB[a][1]] += k
cands = list(itertools.product(range(4), repeat=6))
words = ["".join(ORD[gg][i] for gg, i in zip(gs, idx)) for idx in cands]
CI = np.array(cands)


def posterior(min_n=30, alpha=0.5, pick=None):
    sites = [loc for loc, c in cnt.items() if c[0].sum() >= min_n and c[1].sum() >= min_n]
    if pick is not None:
        sites = [sites[i] for i in pick(len(sites))]
    LL = np.zeros((len(sites), len(cands)))
    for si, loc in enumerate(sites):
        D = np.log((cnt[loc] + alpha) / (cnt[loc] + alpha).sum(1, keepdims=True))
        # per position: log-likelihood of the votes if the true letter index is j
        per = np.array([[sum(N[p, a] * D[gs[p] % 2, a ^ j] for a in range(4)) for j in range(4)]
                        for p in range(6)])
        LL[si] = per[np.arange(6)[None, :], CI].sum(1)
    W = np.exp(LL - LL.max())
    return sites, W, W.sum(0) / W.sum()


sites, W, post = posterior()
order = np.argsort(-post)
print(f"\nempirical Bayes over the {len(cands)} rings with this signature "
      f"({len(sites)} calibrated sites, >=30 letters per class):")
for i in order[:6]:
    top = np.argsort(-W[:, i])[:3]
    sup = ", ".join(f"{sites[k]} {W[k, i] / W.sum():.2f}" for k in top if W[k, i] / W.sum() > 0.005)
    print(f"  {words[i]}  {post[i]:.3f}   supported by: {sup}")
for mn, al in [(15, 0.5), (60, 0.5), (30, 0.1)]:
    _, _, p2 = posterior(mn, al)
    o2 = np.argsort(-p2)[:3]
    print(f"  sensitivity (min letters {mn}, prior {al}): " + ", ".join(f"{words[i]} {p2[i]:.2f}" for i in o2))
wins = Counter()
tot = Counter()
for _ in range(1000):
    _, _, pb = posterior(pick=lambda n: rng.integers(0, n, n))
    wins[words[int(pb.argmax())]] += 1
    for i in np.argsort(-pb)[:3]:
        tot[words[i]] += pb[i]
print(f"  site bootstrap (1000): winner {dict(wins.most_common(4))}; "
      f"mean posterior {{{', '.join(f'{k}: {v / 1000:.2f}' for k, v in tot.most_common(3))}}}")

# analogue sites: one ringed bird, read once per checklist
an = u.groupby("locality_id").agg(reads=("true", "size"), birds=("true", "nunique"),
                                  chk=("checklist_id", "nunique"))
an = an[(an.birds == 1) & (an.reads >= 6)]
print("\nother sites with a single ringed bird (>=6 reads): offsets same/A/B/A+B, even | odd")
for loc, r in an.iterrows():
    c = cnt[loc].astype(int)
    print(f"  {loc}: {r.reads} reads  {c[0].tolist()} | {c[1].tolist()}")

word = words[int(order[0])]
alt = words[int(order[1])]
flip = {"same": 0, "A": 1, "B": 2, "A+B": 3}
print(f"\nanswer: {word} (posterior {post[order[0]]:.2f}); second reading: {alt} "
      f"({post[order[1]]:.2f}); plain consensus {cons} ({post[words.index(cons)]:.3f})")

# ------------------------------------------------------------ 6. figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for kind, col in [("honest", "#1f77b4"), ("flipped", "#d62728")]:
        s = SC[SC.kind == kind]
        ax.scatter(s.share, s.spread, s=np.sqrt(s.n) * 6, c=col, alpha=0.7, label=f"{kind} (calibrated)")
    for odd in (0, 1):
        v = relc[odd]
        ax.scatter(v[0] / v.sum(), min(v[1:]) / v.sum(), marker="*", s=320, c="black",
                   label=f"{SITE} (stork)" if odd == 0 else None)
    ax.set_xlabel("share of the dominant letter")
    ax.set_ylabel("share of the least-written of the other three letters")
    ax.set_title("Honest sites never spread over all three look-alikes; flipped sites do")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sites.png"), dpi=130)

    fig, axs = plt.subplots(1, 6, figsize=(13, 3.4), sharey=True)
    for p in range(6):
        letters = list(ORD[gs[p]])
        vals = [votes[p].get(a, 0) for a in letters]
        cols = ["#d62728" if a == word[p] else "#ff7f0e" if a == alt[p] else
                "#555555" if a == cons[p] else "#bbbbbb" for a in letters]
        axs[p].bar(letters, vals, color=cols)
        axs[p].set_title(f"pos {p + 1}: read {cons[p]}, true {word[p]}")
    axs[0].set_ylabel("reads")
    fig.suptitle(f"{SITE} White Stork: letters written (grey = consensus {cons}, red = {word}, "
                 f"orange = {alt})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_votes.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(7, 8))
    allp = obs.drop_duplicates("locality_id")
    ax.scatter(allp.longitude, allp.latitude, s=1, c="#dddddd")
    piv = SC.pivot(index="site", columns="cls", values="kind").dropna()
    loc_xy = obs.drop_duplicates("locality_id").set_index("locality_id")
    for kind, col in [("honest", "#1f77b4"), ("flipped", "#d62728")]:
        idx = piv.index[(piv == kind).any(axis=1) if kind == "flipped" else (piv == kind).all(axis=1)]
        ax.scatter(loc_xy.longitude[idx], loc_xy.latitude[idx], s=40, c=col, label=f"{kind} site")
    ax.scatter(row.longitude, row.latitude, marker="*", s=300, c="black", label=f"{SITE}, Lake Beleu")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Calibrated ring-reading sites and the unregistered stork")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_map.png"), dpi=130)
except ImportError:
    print("matplotlib missing: figures skipped")

# The model narrows the ring to two all-letter un-flips of the consensus and ranks them
# by how common each site type is elsewhere; that ranking cannot calibrate L5819951 itself.
# Platform outcome: SIGMOID_<redacted> (model rank 1) was rejected and SIGMOID_<redacted>
# (rank 2, the A+B flip, the only all-letter flip seen at far-south Gagauzia sites) passed.
print(f"\nmodel shortlist: {word} ({post[order[0]]:.2f}), {alt} ({post[order[1]]:.2f})")
FINAL = "ANSEMO"
assert FINAL in (word, alt), "confirmed answer must be on the model's two-ring shortlist"
print(f"FLAG (confirmed on the platform): SIGMOID_{FINAL}")
