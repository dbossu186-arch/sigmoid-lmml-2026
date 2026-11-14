# United Kingdom — Call My BLEUff (Medium, Classic ML & Data Science)

## Storyline
On 17 October 1965, BBC2 aired the first Call My Bluff. One team was handed an obscure dictionary word and gave three definitions with perfectly straight faces: one true, two invented. The other team had to say which was which. For more than two decades, most of them with Robert Robinson in the chair, it taught British viewers that the most confident answer is very often the bluff.

Tonight the game is recorded again, in Cambridge, with a twist. The cards carry venues, not words: The Eagle, the pub where Francis Crick is said to have announced in 1953 that he and James Watson had found the secret of life, or Fitzbillies, selling Chelsea buns since 1920. Each panellist describes the venue. One tells it straight, two bluff.

To save the cost of a chairman, the producers gave the job to BLEU, a metric IBM researchers presented in 2002 for grading machine translation. BLEU does not listen; it counts the word sequences a text shares with reference texts. The panellists worked that out long before the producers did.

Take the chair, and call their bluff.

## Objective
Twenty-four rounds of a panel game were recorded. In every round three panellists were handed the same 40 venue cards and each wrote one description per card. For every card the venue guide's own writers had already written five reference descriptions.

In each round exactly one panellist is honest: every one of their 40 descriptions is a genuinely good description of the venue. The other two are bluffing. A bluffer does not bluff on every card: some bluff on most of their cards, some on only a handful, and they do not all bluff the same way, but all of them bluff for the same reason: the show's house scorer is BLEU, and their descriptions are written to please it rather than the listener.

The producers crowned the panellist with the highest BLEU in each round.

Your job:
- Work out how the bluffers game BLEU.
- Propose your own metric or check that catches them.
- Show that it separates the honest decks from the bluffing ones, and use it to name the honest panellist of every round.

Honesty will lead you to the flag.

## Flag
SIGMOID_<WORD>, uppercase, where <WORD> has 8 letters.

How the word is spelt:
- the seat of the honest panellist gives one base-3 digit per round: seat 1 → 0, seat 2 → 1, seat 3 → 2;
- rounds are read in order, three rounds per letter: rounds 1–3 give the first letter, rounds 4–6 the second, and so on;
- three digits d1 d2 d3 give the number 9·d1 + 3·d2 + d3, and 0 → A, 1 → B, … 25 → Z.

## Input (data/)
- references.csv — 4,800 rows: round (1–24), card (1–40), reference. Five reference descriptions per card.
- panel.csv — 2,880 rows: round, card, seat (1–3), text. One description per panellist per card. A panellist's deck is their 40 descriptions in one round.
- bleu.py — the house scorer, in pure Python: corpus-level BLEU-4 over a deck, five references per card. Read it; its configuration matters.
- baseline_solution.py — loads the data, scores every deck with the house scorer and spells the word the producers got.

## Expected output
Submit on the platform:
- The flag;
- The URL to your materials proving the understanding of the gaming mechanisms you identified, the metric or check you propose, the evidence how you found the flag, etc.

## Scoring
- Correct flag, and the linked work shows how you solved the problem: 200 points (PASS)
- Incorrect flag, or a flag without evidence: 0 points (FAIL)

Hints 1–3 exist (locked; unlocking carries a penalty — do not unlock).

## Data link
https://drive.google.com/drive/folders/1Ycd99_NLLKqwjW80B3sca9KGNodF2pNW
