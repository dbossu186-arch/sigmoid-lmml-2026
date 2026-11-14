"""Independent cross-check (not used by solve.py): provenance of every description.

The venue cards, references and panel texts are drawn from the public E2E NLG
Challenge corpus (Novikova et al., 2017; https://github.com/tuetschek/e2e-dataset).
All 4,800 references are verbatim E2E human references.  A panel description
that is also a verbatim E2E human reference *for the same meaning
representation* was written by a person for that venue; one that is not has
been edited (cut, glued, shuffled ...).

    python -I validate_e2e.py /path/to/e2e-dataset     # dir with trainset.csv, devset.csv, testset_w_refs.csv

Prints, per deck, how many of the 40 descriptions are untouched E2E texts and
how the listener check of solve.py agrees with that, card by card.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solve  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9£]+', ' ', s.lower()).strip()


def main(e2e_dir: Path) -> None:
    mrs = defaultdict(set)
    for name in ('trainset.csv', 'devset.csv', 'testset_w_refs.csv'):
        with open(e2e_dir / name, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                mrs[norm(row['ref'])].add(row['mr'])
    refs, panel = solve.load()
    card_mr = {k: Counter(m for r in v for m in mrs[norm(r)]).most_common(1)[0] for k, v in refs.items()}
    assert all(c == 5 for _, c in card_mr.values()), 'every card: all 5 references share one E2E MR'

    tab = Counter()
    print('round   untouched E2E texts per seat   honest(E2E)  honest(listener)')
    agree = 0
    for rnd in sorted({r for r, _ in panel}):
        n, fails = {}, {}
        for seat in (1, 2, 3):
            deck = panel[(rnd, seat)]
            n[seat] = 0
            fails[seat] = 0
            for c, text in deck.items():
                genuine = card_mr[(rnd, c)][0] in mrs.get(norm(text), set())
                fail = solve.listener_check(text, refs[(rnd, c)])['fail']
                n[seat] += genuine
                fails[seat] += fail
                tab[(genuine, fail)] += 1
        h_e2e = max(n, key=n.get)
        h_lis = min(fails, key=fails.get)
        agree += h_e2e == h_lis
        print(f'{rnd:5d}   {n[1]:6d} {n[2]:6d} {n[3]:6d}               seat {h_e2e}       seat {h_lis}')
    print(f'\nrounds where both agree: {agree}/24')
    print('card level (E2E-untouched vs listener check):')
    print(f'  untouched & pass {tab[(True, False)]:5d}   untouched & fail {tab[(True, True)]:5d}')
    print(f'  edited    & pass {tab[(False, False)]:5d}   edited    & fail {tab[(False, True)]:5d}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
