# FRANCE: Point par point (Easy, Data Science & Classic ML)

## Storyline
Paris, 1886. Two French painters, Georges Seurat and Paul Signac, stop mixing colours on the palette. They open a completely new direction instead. They paint with tiny dots of pure colour, placed side by side. Up close, the canvas is only noise. Step back, and the eye joins the dots into a Sunday afternoon by the Seine. The style gets a name: pointillism.

140 years later, a Parisian artist known only as Le Pointilliste picks up the brush to create. This time the canvas is not linen. It is Paris itself, seen through the places where its visitors stay.

Every listing in the city became a dot. Then the artist slipped in some dots of their own. Their colours came from what every traveller weighs before booking a night. Together they form letters.

The artwork was unveiled as a plain spreadsheet, with a single caption: "Grey is pretty. It holds multiple secrets."

## Objective
Find the word the artist hid in the accommodation data of Paris. Separate the letters from the rest of the city, look at each of them, and read the word in the artist's order.

## Flag
SIGMOID_<WORD>, where <WORD> is the string extracted after successfully solving the task.

## Task input
About 28,000 rows, one per listing (a place to stay in Paris), and 18 columns:
- listing_id: listing identifier
- neighbourhood: Paris arrondissement name
- room_type, property_type: kind of accommodation
- latitude, longitude: location of the listing (shifted by up to about 150 m for privacy)
- accommodates, bedrooms, beds: capacity
- price: price per night, EUR
- minimum_nights: minimum stay
- availability_365: nights available for booking in the next 365 days
- number_of_reviews, number_of_reviews_ltm: reviews in total / in the last 12 months
- reviews_per_month: average number of reviews per month since the first review
- review_scores_rating: average guest rating, 0–5
- host_is_superhost: whether the host has Superhost status
- calculated_host_listings_count: number of listings the host has in Paris

## Expected output
Submit the flag in the format SIGMOID_<WORD>, plus a URL with the code and results that helped identify and extract it.

## Scoring
This task uses Flag Evaluation, and organisers review the linked write-up manually.
- Correct flag plus a write-up whose code, figure and labels.csv show the clustered word: 100 points (PASS).
- Incorrect flag, or a correct flag without code, figure and labels that support it: 0 points (FAIL).
