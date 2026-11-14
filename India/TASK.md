# INDIA: Twins Separated at Birth (Easy)

## Storyline
India's monuments have stood for centuries, preserving stories in stone, marble, and intricate architecture. Today, those stories also live inside an ever-growing digital heritage archive containing thousands of photographs collected from across the country, from ancient caves and majestic forts to iconic gateways, temples, palaces, and memorials.

Over time, the archive has become increasingly difficult to organise. Some files originate from the exact same photograph, even though they may no longer appear identical at first glance. At the same time, many different photographs capture the same monument from similar viewpoints, under similar lighting conditions, or only moments apart.

This creates a challenging mix of true duplicates and photographs that are visually similar but represent distinct captures. Your mission is to restore order to the archive by identifying every pair of images that originates from the exact same source photograph, without being misled by different photographs that simply happen to show the same monument.

## Objective
Given a shuffled collection of photographs of Indian monuments, identify all unordered pairs of images that originate from the same source photograph.

Two images showing the same monument are not necessarily duplicates. Different photographs of the same landmark must be treated as distinct images, even when they show similar viewpoints, composition, or lighting. Not every image is guaranteed to have a duplicate.

Recover as many true duplicate pairs as possible while avoiding false matches.

## Flag
The flag follows the format SIGMOID_<PHRASE>. It is revealed only when the submitted solution reaches the required evaluation threshold.

## Task input
Participants receive:
- train-input.zip: the training images, available through the Google Drive link in the task;
- train-output.csv: the known duplicate pairs, included in the task folder.

SHA-256 of train-input.zip: `2e1a5df7bd6c87c581dfdd890ff6d74ebd24dad7a4072d333c4212c6f061316a`

During evaluation, main.py runs in a working directory that already contains the hidden test archive as test-input.zip, which is the file your program must open. It must create output.csv in the same directory. For local practice, rename your train-input.zip to test-input.zip.

## Expected output
Submit a Python program named main.py. During evaluation it processes the input archive and must create output.csv in the same working directory, with exactly these columns: `image_a,image_b`.

Each row is a pair of images predicted to originate from the same source photograph.
- Pairs are unordered: (image_a, image_b) is equivalent to (image_b, image_a).
- Repeated or reversed duplicate rows receive no additional credit.
- Self-pairs such as (image_a, image_a) are invalid.
- Image names must refer to files in the evaluation archive.
- If no duplicate pairs are predicted, output.csv must still contain the header row.

## Scoring
The submitted pairs are compared with the hidden expected pairs to compute precision, recall and F1.
- F1 ≥ 0.88: 100 points (PASS), flag revealed.
- F1 < 0.88: 0 points (FAIL).
