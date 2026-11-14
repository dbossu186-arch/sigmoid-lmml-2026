# Kazakhstan — "Stamp of Disapproval" (Hard, Computer Vision) — Sigmoid LMML 2026

## Storyline
Kazakh postage stamps mix felt ornaments, a dombra, a rocket leaving Baikonur, and Soviet-poster lettering. Aigerim appraises collections for a museum in Almaty. Five stock books came from a retired postal clerk, and she noticed a denomination that never existed: somebody had been "improving" the collection, one stamp per page. She marked each spot on tracing paper and held the sheets of one album up to the lamp.

## Objective
Every album page shows three Kazakh postage stamps. On each page exactly one stamp was altered in one small region: a digit changed, an element cloned, or an inscription removed. Find that region on every page. Mark the regions of all pages of one album on a single sheet, and they draw a letter. Comparing a stamp with a copy found elsewhere (catalogue, internet) or checking manually is not detection and doesn't count.

## Flag
SIGMOID_<WORD>, where <WORD> is the retrieved flag.

## Input
JPEG scans of stock-book pages, 1600×1200:
- albums/album_1 … album_5: 26 pages each. Albums are in letter order, so 5 albums give 5 letters. File names carry no meaning.
- exhibits/: 30 single stamps altered in the same ways, each with a mask (exhibit_NN.jpg, exhibit_NN_mask.png; white = altered). None of them appears on an album page.
- No labelled album pages and no untouched copies of the altered stamps.
- baseline_solution.py shows the input and output formats.

Dataset: https://drive.google.com/drive/folders/1VKPTZUZf9sUxQrBkpr4kuDpcAXRfKZoC (ID 1VKPTZUZf9sUxQrBkpr4kuDpcAXRfKZoC)

## Expected output / scoring
Submit the flag plus a URL with the detection code and all materials, including outputs (boxes, overlay sheets, write-up). Correct flag with supporting material: 300 points; otherwise 0. 3 hints, locked (penalty).
