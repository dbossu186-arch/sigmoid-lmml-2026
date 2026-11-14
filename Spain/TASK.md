# SPAIN: ¡Feliz Navidad! (Easy)

## Storyline
It is December 2025, and the 21 pilots of a small charter company have a Christmas mission: bring home authentic Spanish souvenirs. Flying between cities is their day job, so each pilot hops across Spain and buys every specialty in the city it comes from.

The head office has seen this go wrong before. Tourist traps charge three times the going rate, and stalls selling "saffron" at a tenth of the price are selling something else entirely. So the office keeps a recommended list of six souvenirs: Jamon, Churros, Turron, Paella, Saffron, Barquillos.

For each item, office_price_guide.csv gives its number on this list (List_no), its home city and the fair price range per unit a pilot should expect to pay.

Office rule: a souvenir counts only if its price per unit falls inside the fair range on the list. Anything cheaper is a fake, anything more expensive is a tourist trap, and neither makes it onto the board.

Every January the office pins up its Christmas board. Each pilot gets a column, where HR checks which pilot brought a given souvenir for a fair price. This year, HR says, the board looked unusual.

The pilots have handed in every receipt, and there are a lot of them. Rebuild the board.

## Objective
Apply the office rule to the receipts and visualise the right relationship between the columns so that you can reconstruct the office board. The board spells a word, and that word is the flag.

## Flag
SIGMOID_<WORD>, where <WORD> is the word you read on the board, in uppercase letters A–Z. Some glyphs on the board might be lowercase-shaped; submit them uppercase anyway.

## Task input
**office_price_guide.csv:** the office's recommended list, 6 rows.

| Column | Meaning |
|---|---|
| List_no | Number of the item on the office's list, 1–6 |
| Product | Souvenir name (same spelling as in purchases.csv) |
| Unit | The unit the fair price refers to: kg, dozen, bar, kit, g, box |
| Fair_min_eur, Fair_max_eur | Fair price range in EUR per unit, inclusive |
| Home_city | Where the souvenir comes from |

**purchases.csv:** one row per receipt, 577 rows and 14 columns.

| Column | Meaning |
|---|---|
| Receipt_id | Receipt identifier, in chronological order |
| Date | Purchase date (1–22 December 2025) |
| Pilot_no | Pilot number, 1–21. All 21 pilots have a column on the board |
| City | City where the purchase was made |
| Shop_type | Market stall, specialty store, supermarket, souvenir shop or airport shop |
| Product | Souvenir bought |
| Quantity | Amount bought, in the unit given by the office guide |
| Total_paid_eur | Total amount paid on the receipt, EUR |
| Payment | Card, cash or Bizum |
| Shop_rating | Online rating of the shop, 1–5 |
| Flight_km, Flight_min | Distance and duration of the flight that brought the pilot to that city |
| Temperature_c | Temperature in the city at the time of purchase, °C |
| Luggage_kg | Weight of the pilot's bag after the purchase |

Environment: Python 3.10–3.13 with numpy==2.2.6, pandas==2.3.3 and matplotlib==3.10.9; notebook/ipykernel only for baseline_solution.ipynb.

## Expected output
Submit the flag SIGMOID_<WORD> and a URL (GitHub, Google Drive, Colab, ...) to the notebook or script you used and the resulting figure, in which the word is visible.

## Scoring
This task uses Flag Evaluation, and organisers review the linked write-up manually.
- Correct flag plus a write-up whose code produces the figure showing the word: 100 points (PASS).
- Incorrect flag, or a correct flag without supporting code and figure: 0 points (FAIL).
- Three hints are available; each one revealed deducts its penalty from the task's points.
