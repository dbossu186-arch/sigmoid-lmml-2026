# Tanzania — "Please, You Are So Delusional!"

**Flag:** printed by `python retrieval_pipeline.py` (SHA-256 of the retrieved documents' verification characters).

## Reproduce
```
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python numpy==1.26.4
.venv/bin/python retrieval_pipeline.py      # writes retrieved_ids.json, prints flag
.venv/bin/python check_retrieval.py         # writes review.md
```
`corpus.jsonl`, `questions.json`, `requirements.txt` are symlinks into `work/` (the extracted, unmodified zip).
`flag_utils.py` is an unmodified copy of `work/flag_utils.py` (checked with `cmp`). Only numpy is needed;
the repaired pipeline no longer downloads sentence-transformers models.

## Data
- 744 docs: 600 generic venue bulletins (`TZ-G-*`, no collection info), and 24 exhibits x 3 cities
  (Dar es Salaam / Arusha / Mwanza), each with a `CURRENT ACCESS NOTICE` (`TZ-kk-jA`, holds the
  collection desk + time) and an `ARCHIVED BACKGROUND` decoy (`TZ-kk-jB`, near-identical wording,
  says "does not replace the current access notice", no operational details).
- 72 questions: each asks for the current access notice of one exhibit (formal name + an informal alias)
  in one city.

## Bugs found in the starter `retrieval_pipeline.py` (and fixes)
1. **Two different encoders**: documents embedded with `all-MiniLM-L6-v2`, queries with `e5-small-v2`.
   Both are 384-d so no error is raised, but the dot products compare unrelated vector spaces (garbage scores).
   Fix: a single shared vector space for queries and documents (TF-IDF, one vocabulary).
2. **e5 prefix misuse**: queries get `query:` but documents never get `passage:` (moot after fix 1).
3. **Fixed 180-character chunking with max-over-chunks**: slices cut mid-sentence, separating the
   exhibit name, city and collection details, so a boilerplate chunk shared by many docs
   ("Similar programme activities may run in other cities...") can win for the wrong document.
   Fix: index whole documents (title + body).
4. **No notice-status awareness**: archived background decoys literally contain "current access notice"
   and are shorter, so they score *higher* lexically (pure TF-IDF picks the right doc for only 3/72 questions;
   lexical top-1 is the archived decoy). Fix: reranking bonus for `CURRENT` titles, penalty for
   `ARCHIVED/EXPIRED/SUPERSEDED/WITHDRAWN`.
5. **No city constraint**: the same exhibit appears in three cities with identical text apart from
   venue/date/desk/time. Fix: city parsed from the question must match the document's city.
6. **No exhibit constraint / alias confusion**: questions use informal aliases ("sunlight colour station");
   fix: exact formal exhibit name (from corpus titles) must match.
7. Tokenisation / determinism: proper lowercase tokeniser with stopwords, positive smoothed IDF, cosine
   on L2-normalised vectors, argmax, ties broken by corpus order (no ties occurred).

## Validation
`check_retrieval.py` -> `review.md`: for every question the top-3 docs with scores, the lexical-only top-3,
and an independent structural check parsed from the document **body** (status CURRENT, exhibit and city in
the "reserved the X for the Y stop" sentence, venue city, desk and time present, and that it is the unique
such document). Result: 72/72 pass, 0 low-margin cases (smallest margin between #1 and #2 = 1.90;
the runner-up is always the archived decoy or the same exhibit in another city).
All answers follow the pattern q(3(k-1)+j) -> TZ-kk-jA (exhibit k, city j).
