# Egypt — Always Valid, No Excuses (Easy, General AI)

## Storyline
Historians in Cairo have run OCR over thousands of old Egyptology books and pulled out record cards, one per figure or monument, as plain key: value text. The digital catalogue accepts only strictly valid JSON: one malformed card and it is rejected. You are handed the museum's small talking oracle, a local model (Qwen3-1.7B), and told: use the oracle, not your own hand. Left to itself the oracle drifts: it quotes a number, forgets a missing field, mangles a backslash, keeps a torn page number. Hold it to the format until all 100 cards are flawless. 99 is a failure.

## Objective
Write a main.py that uses the provided local model to convert each OCR'd record into one line of strictly valid JSON carrying exactly the same data. main.py reads test-input.txt from its own directory (one record per line, UTF-8) and writes test-output.txt beside it, where line N is the JSON for input line N. Non-interactive and offline.

## Flag
SIGMOID_<WORD>, released by the evaluator only when all 100 hidden records are correct.

## Input
Sample data: LMML_Egypt_TRAIN_DATA.zip (data/). Example sample (not training data, there is nothing to train): train-input.txt (OCR records, one per line) and train-output.txt (one valid JSON line per record). The hidden test set uses different values: match the format, not the sample data. (The Drive folder also contains data/test-input.txt.)

Each record is key: value fields separated by |. It is not JSON, and it is messy in two ways:
- Messy spacing around : and |, and at the ends. Trim each key and value.
- Scan noise that is not a field and must be dropped: page numbers (page 22), headers (Chapter IX), footers (- 45 -), bracketed marks ([torn], [illegible], (margin)), figure labels, garble (###). Only a segment shaped like key: value with a simple word key is a real field.

Example:
Input : `page 22| king :Menes| title: ruled with "the Aten" |[torn]| alive :true | Chapter IX`
Output: `{"king":"Menes","title":"ruled with \"the Aten\"","alive":true}`

Conversion rules for each real field (after trimming):
- empty value → null
- exactly true / false → JSON booleans
- whole number (40, 0, -3) → JSON number
- anything else → JSON string, with " and \ escaped. Only the first : separates key from value (ratio: 3:1 → "ratio":"3:1").

## Model / environment (pinned)
Model: Qwen/Qwen3-1.7B-GGUF (~1.8 GB, not bundled; https://huggingface.co/Qwen/Qwen3-1.7B-GGUF). Download once into models/ next to main.py, then run offline:

    huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models

Environment: Python 3.10–3.12 with huggingface-hub==0.26.2 and llama-cpp-python==0.3.35 built with CUDA (load with n_gpu_layers=-1). See requirements.txt for the build command. Evaluation runs on GPU with a 60-second time limit.

## Expected output
Submit test-output.txt (upload, .txt, max 10 MB): one strict JSON object per input record, holding every real field typed by the rules above, with scan noise dropped and nothing added. Key order and whitespace do not matter (the grader compares parsed data, not bytes).

## Scoring
Each of the 100 hidden records passes iff its output line parses as strict JSON and decodes to exactly the expected data.
- 100/100 records pass: 100 points (PASS, flag released)
- 99 or fewer: 0 points (FAIL)

Hints 1–3 exist (locked; penalty — do not unlock).

## Data link
https://drive.google.com/drive/folders/19F3s6K-n3GlN3Q4k0_wnVML_CbjX5fYZ
