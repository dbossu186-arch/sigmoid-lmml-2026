# Tanzania — "Please, You Are So Delusional!" (Easy, 100 pts)
Safari ya Sayansi's digital assistant often sends visitors to the wrong operational notice. Repair the retrieval code so every question points to the correct source document.
Objective: repair the supplied retrieval pipeline so it returns the best source document for every question; use the results to recover the flag.
Flag: run `python retrieval_pipeline.py`; it writes retrieved_ids.json and prints a candidate flag. flag_utils.py reads each retrieved document's verification character, concatenates them in questions.json order, SHA-256 hashes the sequence, and formats the first 20 uppercase hex chars as SIGMOID_<WORD>. Submit the flag printed by the repaired pipeline.
Input: corpus.jsonl (source docs), questions.json (in order), retrieval_pipeline.py (starter to repair), flag_utils.py (supplied helper), requirements.txt. Python 3.11; `python -m pip install -r requirements.txt`.
Expected output: retrieved_ids.json — JSON list, exactly one result per question, in questions.json order: [{"question_id": "q001", "document_id": "TZ-01-1A"}]. Submit the flag + a link to the repaired code and retrieved_ids.json.
Scoring: PASS = correct flag + reviewable output that reproduces it.
Data: https://drive.google.com/drive/folders/1E94rYgT1d4vum8zPRHr-TMATz4NxJssb
