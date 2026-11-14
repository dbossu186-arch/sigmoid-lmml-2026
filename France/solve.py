"""Point par point - find the word hidden in paris_listings.csv.

Pipeline
1. Detect planted rows. Every letter is painted with a smooth "colour"
   gradient made of three traveller features: price, availability_365 and
   reviews_per_month. Real listings have no such joint structure, so we score
   each row by how many neighbours it has in the joint space
   (colour + location) versus the same count after shuffling colours across
   locations. The score histogram is bimodal; the high mode holds the planted dots.
2. Split the planted dots into letters with HDBSCAN in the same joint space,
   then merge sub-clusters that touch (single-linkage gap < 2), which leaves
   6 letters of about 230-340 dots each.
3. Look at each letter. Its three colour values (standardised log price,
   availability_365, log reviews_per_month) lie on a flat 2D plane (3rd PCA
   component ~1-2% of variance). Projecting a letter onto that plane, then
   whitening it, shows the glyph without the 150 m privacy jitter on the map.
   PCA axes have no built-in "up", so each letter is rotated/mirrored for
   display (DISPLAY_ORIENT below); this only turns the picture, it does not
   change which dots belong to which letter.
4. The artist's order is the letters' price level (cheapest first): N E U R O N.

Usage: python solve.py   (needs pandas, numpy, scikit-learn, matplotlib)
Outputs: labels.csv, pointillism_solution.png
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KDTree

SEED = 0
SCORE_THRESHOLD = 4.5  # valley of the bimodal log2(score) histogram
READ_LETTERS = "NEURON"  # letters as read from the figure, in price order
# Display orientation per letter (index in price order): 0-3 = rotate by k*90 deg,
# 4-7 = same rotation then mirror left-right. Chosen by eye so each glyph reads upright.
DISPLAY_ORIENT = {0: 2, 1: 3, 2: 4, 3: 0, 4: 0, 5: 0}
ROT = [np.eye(2), np.array([[0, -1], [1, 0]]), -np.eye(2), np.array([[0, 1], [-1, 0]])]


def colour_plane(s):
    """Whitened 2D coordinates of a letter's dots in its own colour plane."""
    Y = np.c_[np.log(s.price), s.availability_365, np.log(s.reviews_per_month)]
    Y = (Y - Y.mean(0)) / Y.std(0)
    keep = np.ones(len(s), bool)
    for _ in range(4):  # trim dots that sit off the plane (stray real listings)
        c = Y[keep].mean(0)
        _, _, vt = np.linalg.svd(Y[keep] - c, full_matrices=False)
        off = np.abs((Y - c) @ vt[2])
        keep = off < np.quantile(off[keep], 0.9) * 1.3
    vt = vt * np.sign(vt[np.arange(3), np.abs(vt).argmax(1)])[:, None]  # fixed signs
    Z = ((Y - c) @ vt[:2].T)[keep]
    w, U = np.linalg.eigh(np.cov(Z.T))
    Z = Z @ (U @ np.diag(w ** -0.5) @ U.T)
    return Z[np.linalg.norm(Z, axis=1) < 3.2]  # drop the few far outliers


def orient(Z, o):
    W = Z @ ROT[o % 4].T
    return W * [-1, 1] if o >= 4 else W

df = pd.read_csv("paris_listings.csv")

# --- 1. joint colour + location space -------------------------------------
colour = np.c_[
    np.log(df.price) / 0.12,
    df.availability_365 / 25,
    np.log(df.reviews_per_month) / 0.15,
]
space = np.c_[df.latitude * 111, df.longitude * 73] / 0.6  # km / 0.6
joint = np.c_[colour, space]

radius = 1.0
observed = KDTree(joint).query_radius(joint, radius, count_only=True) - 1
rng = np.random.default_rng(SEED)
expected = np.zeros(len(df))
n_perm = 10
for _ in range(n_perm):
    perm = rng.permutation(len(df))
    shuffled = np.c_[colour[perm], space]
    expected += KDTree(shuffled).query_radius(joint, radius, count_only=True)
expected /= n_perm
df["score"] = (observed + 1) / (expected + 1)
df["planted"] = (df.score > SCORE_THRESHOLD).astype(int)
print(f"planted dots: {df.planted.sum()} of {len(df)}")

# --- 2. letters ------------------------------------------------------------
fake = df.planted.values == 1
Xf = joint[fake]
sub = HDBSCAN(min_cluster_size=40, min_samples=5).fit_predict(Xf)
ids = [k for k in np.unique(sub) if k >= 0]
adj = np.zeros((len(ids), len(ids)), bool)
for i, a in enumerate(ids):
    for j, b in enumerate(ids):
        if i < j and cdist(Xf[sub == a], Xf[sub == b]).min() < 2.0:
            adj[i, j] = adj[j, i] = True
_, comp = connected_components(adj, directed=False)
letter_of_sub = {a: comp[i] for i, a in enumerate(ids)}
raw = np.array([letter_of_sub.get(k, -1) for k in sub])

# order letters by price level (the artist's order)
tmp = df[fake].assign(raw=raw)
order = (
    tmp[tmp.raw >= 0].groupby("raw").price.median().sort_values().index.tolist()
)
rank = {r: i for i, r in enumerate(order)}
df["letter_idx"] = -1
df.loc[fake, "letter_idx"] = [rank.get(r, -1) for r in raw]
assert len(order) == len(READ_LETTERS), f"found {len(order)} letters"
df["letter"] = df.letter_idx.map(lambda i: READ_LETTERS[i] if i >= 0 else "")

summary = (
    df[df.letter_idx >= 0]
    .groupby(["letter_idx", "letter"])
    .agg(
        dots=("price", "size"),
        median_price=("price", "median"),
        median_availability=("availability_365", "median"),
        median_rpm=("reviews_per_month", "median"),
    )
)
print(summary.round(2))
word = "".join(READ_LETTERS[i] for i in range(len(order)))
print("WORD:", word, "->  FLAG: SIGMOID_" + word)

df[["listing_id", "planted", "letter_idx", "letter", "score"]].round(
    {"score": 3}
).to_csv("labels.csv", index=False)

# --- 3. figure -------------------------------------------------------------
n = len(order)
fig = plt.figure(figsize=(4 * n, 13))
gs = fig.add_gridspec(2, n, height_ratios=[1.6, 1])
ax = fig.add_subplot(gs[0, :])
ax.scatter(df.longitude, df.latitude, s=0.3, c="#cfcfcf", label="real listings")
cmap = plt.get_cmap("tab10")
for i in range(n):
    s = df[df.letter_idx == i]
    ax.scatter(s.longitude, s.latitude, s=4, color=cmap(i),
               label=f"{READ_LETTERS[i]} (~EUR {s.price.median():.0f})")
ax.set_aspect(1 / np.cos(np.deg2rad(48.86)))
ax.set_title("Planted dots on the map (grey = the rest of Paris)")
ax.legend(markerscale=4, loc="lower right")
for i in range(n):
    s = df[df.letter_idx == i]
    W = orient(colour_plane(s), DISPLAY_ORIENT.get(i, 0))
    a = fig.add_subplot(gs[1, i])
    a.scatter(W[:, 0], W[:, 1], s=7, color=cmap(i))
    a.set_aspect(1)
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(f"#{i + 1}: '{READ_LETTERS[i]}'  median price EUR {s.price.median():.0f}")
    a.set_xlabel("colour plane (price, availability, reviews/month)", fontsize=8)
fig.suptitle(f"Point par point  ->  {word}", fontsize=18)
fig.tight_layout()
fig.savefig("pointillism_solution.png", dpi=110)
print("wrote labels.csv and pointillism_solution.png")
