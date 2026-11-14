"""Call My BLEUff -- name the honest panellist of every round.

Pure Python (stdlib only; matplotlib optional for the figure).

    python -I solve.py                # prints the table, the word and the flag
    python -I solve.py --figure       # also writes separation.png

The check ("listener check") is applied card by card.  A description fails
it if a listener would notice any of the four things BLEU cannot see:

  OMIT    it leaves out >= 2 of the card's facts.  A card's facts are the
          attributes (type of venue, food, price, rating, area, family
          friendliness) and the named venues (the venue itself, the landmark
          it is near) that at least half of the five references mention.
          BLEU is precision-only and its brevity penalty is pooled over the
          deck and taken against the *closest* reference, so dropping the
          risky phrases is free.
  REPEAT  it states >= 2 of those attributes twice, in separate sentences
          (two descriptions glued together).  Clipping is per n-gram against
          whichever reference has it most, so a concatenation of two
          reference-like sentences harvests n-grams from several references,
          and the extra length pays the deck's brevity penalty for the
          truncated cards.
  COPY    >= 90 % of it is one contiguous run lifted from a reference
          (verbatim copies, usually cut short).
  TRUNC   it stops on a word that cannot end a sentence ("... rating of 3
          out.", "... near Raja.").

The deck score is the number of failing cards (0..40).  The honest deck is the
one with the fewest; the margin is the gap to the next-best deck.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# --------------------------------------------------------------------------- text
TOK = re.compile(r"\w+|[^\w\s]")


def tok(s: str) -> list[str]:
    return TOK.findall(s.lower())


NEAR = ['raja indian cuisine', 'rainbow vegetarian café', 'crowne plaza hotel', 'crown plaza hotel',
        'express by holiday inn', 'yippee noodle bar', 'all bar one', 'burger king', 'café adriatic',
        'café brazil', 'café rouge', 'café sicilia', 'clare hall', 'the portland arms', 'the sorrento',
        'the bakers', 'avalon', 'ranch', 'the rice boat']
NAMES = ['alimentum', 'aromi', 'bibimbap house', 'blue spice', 'browns cambridge', 'clowns', 'cocum', 'cotto',
         'fitzbillies', 'giraffe', 'green man', 'loch fyne', 'midsummer house', 'strada', 'taste of cambridge',
         'the cambridge blue', 'the cricketers', 'the dumpling tree', 'the eagle', 'the golden curry',
         'the golden palace', 'the mill', 'the olive grove', 'the phoenix', 'the plough', 'the punter',
         'the rice boat', 'the twenty two', 'the vaults', 'the waterman', 'the wrestlers',
         'travellers rest beefeater', 'wildwood', 'zizzi']
ENTS = sorted({e[4:] if e.startswith('the ') else e for e in NEAR + NAMES}, key=len, reverse=True)

ATTR = {
    'eatType': r"coffee|\bcaf[eé]\b|\bpubs?\b|public house|restaurant|diner|eatery|establishment|venue|\bplace\b|\bjoint\b|\bbar\b|\bshop\b|bistro",
    'food': r"japanese|sushi|chinese|english|british|breakfast|fast[- ]?food|burger|fries|takeaway|french|wine|cheese|indian|curry|italian|pasta|pizza|spaghetti|cuisine|\bfood",
    'price': r"pric|cost|cheap|expensive|£|pound|euro|dollar|afford|budget|value|\bspend|money|\bl20|20-25|\b20\b|\b30\b",
    'rating': r"rat(e|ed|es|ing|ings)\b|star|review|out of|excellent|poor|feedback|scor|\b1out|acclaim",
    'area': r"river|cent(re|er)|city|downtown|\btown\b",
    'family': r"famil|\bkids?\b|child|adult",
}


def _strip_ents(text: str) -> str:
    t = text.lower()
    for e in ENTS:                      # "Raja Indian Cuisine" must not count as Indian food
        t = t.replace(e, ' ENT ')
    return t


def attrs(text: str) -> set[str]:
    t = _strip_ents(text)
    return {a for a, p in ATTR.items() if re.search(p, t)}


def ents(text: str) -> set[str]:
    t, found = text.lower(), set()
    for e in ENTS:
        if e in t:
            found.add(e)
            t = t.replace(e, ' ')
    return found


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.split()) >= 2]


def card_facts(refs: list[str]) -> tuple[set[str], set[str]]:
    """Facts mentioned by at least half of the references."""
    ac, ec = Counter(), Counter()
    for r in refs:
        ac.update(attrs(r))
        ec.update(ents(r))
    n = len(refs)
    return {a for a, c in ac.items() if 2 * c >= n}, {e for e, c in ec.items() if 2 * c >= n}


def missing_facts(text: str, facts) -> list[str]:
    A, E = facts
    return sorted((A - attrs(text)) | (E - ents(text)))


def repeated_facts(text: str, facts) -> list[str]:
    A, _ = facts
    c = Counter()
    for s in sentences(text):
        c.update(attrs(s) & A)
    return sorted(k for k, v in c.items() if v >= 2)


def longest_common_run(a: list[str], b: list[str]) -> int:
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b):
            if x == y:
                cur[j + 1] = prev[j] + 1
                if cur[j + 1] > best:
                    best = cur[j + 1]
        prev = cur
    return best


DANGLING = set("""a an the of and or but nor is are was were be been to in on at by for from with near into than that
which who whose its their his her this these those has have had out more less very also called named customer coffee
kid child provides serves offers located situated""".split())
MULTI = [e for e in NEAR + NAMES if ' ' in e]


def truncated(text: str) -> bool:
    w = [x for x in tok(text) if x.isalnum()]
    if not w or w[-1] in DANGLING:
        return True
    tail = ' ' + ' '.join(w[-3:])
    if any(tail.endswith(' ' + n) for n in NEAR + NAMES):
        return False
    for e in MULTI:                      # stops inside a venue name: "... near Raja."
        ew = e.split()
        if any(tail.endswith(' ' + ' '.join(ew[:k])) for k in range(1, len(ew))):
            return True
    return False


# --------------------------------------------------------------------------- check
CHECKS = ('OMIT', 'REPEAT', 'COPY', 'TRUNC')


def listener_check(text: str, refs: list[str], omit_k: int = 2, repeat_k: int = 2, copy_th: float = 0.9,
                   use_trunc: bool = True) -> dict:
    facts = card_facts(refs)
    out, rt = tok(text), [tok(r) for r in refs]
    miss = missing_facts(text, facts)
    rep = repeated_facts(text, facts)
    copy = max(longest_common_run(out, r) for r in rt) / max(1, len(out))
    res = {'OMIT': len(miss) >= omit_k, 'REPEAT': len(rep) >= repeat_k, 'COPY': copy >= copy_th,
           'TRUNC': use_trunc and truncated(text),
           'missing': miss, 'repeated': rep, 'copy_frac': copy}
    res['fail'] = any(res[c] for c in CHECKS)
    return res


# --------------------------------------------------------------------------- BLEU (house scorer, re-implemented)
def house_bleu(outputs: list[str], refsets: list[list[str]]) -> float:
    import math
    m, t, ol, rl = [0] * 4, [0] * 4, 0, 0
    for o, R in zip(outputs, refsets):
        o, R = tok(o), [tok(r) for r in R]
        for n in range(1, 5):
            cnt = Counter(tuple(o[i:i + n]) for i in range(len(o) - n + 1))
            ceil = Counter()
            for r in R:
                for g, c in Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1)).items():
                    ceil[g] = max(ceil[g], c)
            m[n - 1] += sum(min(c, ceil[g]) for g, c in cnt.items())
            t[n - 1] += sum(cnt.values())
        ol += len(o)
        rl += min((abs(len(r) - len(o)), len(r)) for r in R)[1]
    p = [a / b for a, b in zip(m, t)]
    if min(p) == 0:
        return 0.0
    bp = 1.0 if ol >= rl else math.exp(1 - rl / ol)
    return bp * math.exp(sum(math.log(x) for x in p) / 4)


# --------------------------------------------------------------------------- main
def load():
    refs = defaultdict(list)
    with open(DATA / 'references.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            refs[(int(r['round']), int(r['card']))].append(r['reference'])
    panel = defaultdict(dict)                       # (round, seat) -> {card: text}
    with open(DATA / 'panel.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            panel[(int(r['round']), int(r['seat']))][int(r['card'])] = r['text']
    return refs, panel


def decode(seats: list[int]) -> str:
    return ''.join(chr(65 + 9 * (seats[i] - 1) + 3 * (seats[i + 1] - 1) + (seats[i + 2] - 1))
                   for i in range(0, len(seats), 3))


def main(figure: bool = False) -> None:
    refs, panel = load()
    rounds = sorted({r for r, _ in panel})
    per_card = []                                   # for the figure / write-up
    score, bleu, by_check = {}, {}, {}
    for (rnd, seat), deck in sorted(panel.items()):
        cards = sorted(deck)
        res = [listener_check(deck[c], refs[(rnd, c)]) for c in cards]
        score[(rnd, seat)] = sum(r['fail'] for r in res)
        by_check[(rnd, seat)] = {k: sum(r[k] for r in res) for k in CHECKS}
        bleu[(rnd, seat)] = house_bleu([deck[c] for c in cards], [refs[(rnd, c)] for c in cards])
        per_card += [(rnd, seat, c, r) for c, r in zip(cards, res)]

    print('Failing cards per deck (listener check)   |  house BLEU')
    print('round  seat1 seat2 seat3  honest margin    |  seat1 seat2 seat3  BLEU pick')
    honest, bleu_pick = [], []
    for rnd in rounds:
        s = [score[(rnd, k)] for k in (1, 2, 3)]
        b = [bleu[(rnd, k)] for k in (1, 2, 3)]
        h = 1 + s.index(min(s))
        margin = sorted(s)[1] - sorted(s)[0]
        honest.append(h)
        bleu_pick.append(1 + b.index(max(b)))
        print(f'{rnd:5d}  {s[0]:5d} {s[1]:5d} {s[2]:5d}  seat {h}  {margin:4d}     |  '
              f'{b[0]:.3f} {b[1]:.3f} {b[2]:.3f}  seat {bleu_pick[-1]}')
    print('\nBreakdown of failures by check (round: seat1 / seat2 / seat3):')
    for rnd in rounds:
        print(f'  r{rnd:02d}: ' + ' / '.join(
            ' '.join(f'{k}={by_check[(rnd, s)][k]}' for k in CHECKS if by_check[(rnd, s)][k]) or 'clean'
            for s in (1, 2, 3)))
    word = decode(honest)
    print('\nhonest seats :', ''.join(map(str, honest)))
    print('digits       :', ' '.join(''.join(str(s - 1) for s in honest[i:i + 3]) for i in range(0, 24, 3)))
    print("BLEU's word  : SIGMOID_" + decode(bleu_pick), '(the producers, wrong)')
    print('word         :', word)
    print('flag         : SIGMOID_' + word)

    if figure:
        make_figure(rounds, score, bleu, honest, HERE / 'separation.png')


def make_figure(rounds, score, bleu, honest, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    col = {'honest': '#1b7837', 'bluff': '#b2182b'}
    for ax, metric, lab in ((axes[0], bleu, 'house BLEU (higher = "better")'),
                            (axes[1], score, 'cards failing the listener check (of 40)')):
        for rnd, h in zip(rounds, honest):
            for seat in (1, 2, 3):
                v = metric[(rnd, seat)]
                is_h = seat == h
                ax.scatter(rnd, v, s=60 if is_h else 34, marker='o' if is_h else 'x',
                           color=col['honest'] if is_h else col['bluff'], zorder=3)
                ax.annotate(str(seat), (rnd, v), textcoords='offset points', xytext=(5, -3), fontsize=7,
                            color='#555')
        ax.set_ylabel(lab)
        ax.grid(axis='y', alpha=.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].scatter([], [], color=col['honest'], label='honest deck (named by the listener check)')
    axes[0].scatter([], [], color=col['bluff'], marker='x', label='bluffing decks')
    axes[0].legend(frameon=False, loc='upper left', ncol=2)
    wrong = sum(max((1, 2, 3), key=lambda k: bleu[(r, k)]) != h for r, h in zip(rounds, honest))
    axes[0].set_title(f'House BLEU crowns a bluffer in {wrong} of {len(rounds)} rounds; the listener check '
                      'puts the honest deck clear of both bluffers in every round', fontsize=10)
    axes[1].set_xlabel('round  (small numbers next to points = seat)')
    axes[1].set_xticks(rounds)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print('figure       :', path)


def sensitivity() -> None:
    """Re-run the decision under other reasonable thresholds; the honest seats should not move."""
    refs, panel = load()
    rounds = sorted({r for r, _ in panel})
    grid = [dict(omit_k=o, repeat_k=r, copy_th=c, use_trunc=t)
            for o in (2, 3) for r in (2, 3) for c in (0.8, 0.9, 0.95) for t in (True, False)]
    base = None
    for g in grid:
        score = {(rnd, seat): sum(listener_check(deck[c], refs[(rnd, c)], **g)['fail'] for c in deck)
                 for (rnd, seat), deck in panel.items()}
        honest, margins = [], []
        for rnd in rounds:
            s = sorted(((score[(rnd, k)], k) for k in (1, 2, 3)))
            honest.append(s[0][1])
            margins.append(s[1][0] - s[0][0])
        w = decode(honest)
        base = base or w
        print(f"{str(g):75s} word={w} min-margin={min(margins)} {'' if w == base else '  <-- differs'}")


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        sensitivity()
    else:
        main(figure='--figure' in sys.argv)
