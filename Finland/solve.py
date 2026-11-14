"""Sigmoid Finland: clean the hand-logged route log, rebuild the network, rank towns by degree.

Usage:  python -I solve.py [path/to/routes.csv]
"""
import sys, unicodedata, collections
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "routes.csv"


def fold(s: str) -> str:
    """Surface normalisation: whitespace, trailing dot, case, diacritics (ä/ö/å -> a/o/a)."""
    s = s.strip().rstrip(".").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


# Swedish (or alternative) name -> canonical Finnish name.  Keys are already fold()ed.
ALIAS = {
    "helsingfors": "helsinki", "esbo": "espoo", "vanda": "vantaa", "grankulla": "kauniainen",
    "abo": "turku", "tammerfors": "tampere", "uleaborg": "oulu", "vasa": "vaasa",
    "bjorneborg": "pori", "raumo": "rauma", "nystad": "uusikaupunki", "nadendal": "naantali",
    "reso": "raisio", "s:t karins": "kaarina", "virmo": "mynamaki", "pargas": "parainen",
    "kimitoon": "kemionsaari", "villmanstrand": "lappeenranta", "fredrikshamn": "hamina",
    "lovisa": "loviisa", "borga": "porvoo", "sibbo": "sipoo", "kervo": "kerava",
    "tusby": "tuusula", "traskanda": "jarvenpaa", "hyvinge": "hyvinkaa", "vichtis": "vihti",
    "lojo": "lohja", "hogfors": "karkkila", "kyrkslatt": "kirkkonummi", "sjundea": "siuntio",
    "inga": "inkoo", "raseborg": "raasepori", "hango": "hanko", "lahtis": "lahti",
    "tavastehus": "hameenlinna", "birkala": "pirkkala", "ikalis": "ikaalinen", "kumo": "kokemaki",
    "ulvsby": "ulvila", "lundo": "lieto", "s:t michel": "mikkeli", "idensalmi": "iisalmi",
    "kajana": "kajaani", "tornea": "tornio", "brahestad": "raahe", "karleby": "kokkola",
    "jakobstad": "pietarsaari", "nykarleby": "uusikaarlepyy", "larsmo": "luoto",
    "kronoby": "kruunupyy", "korsholm": "mustasaari", "malax": "maalahti", "vora": "voyri",
    "narpes": "narpio", "kristinestad": "kristiinankaupunki", "kasko": "kaskinen",
    "ilmola": "ilmajoki", "lappo": "lapua", "alavo": "alavus",
    "keuru": "keuruu",  # truncated spelling of Keuruu
}


def canon(raw: str) -> str:
    f = fold(raw)
    return ALIAS.get(f, f)


def build(df, by_type=False):
    edges = set()
    for a, b, t in zip(df.A, df.B, df.route_type):
        if a == b:
            continue  # self-loop
        e = tuple(sorted((a, b)))
        edges.add(e + ((t,) if by_type else ()))
    return edges


def degrees(edges):
    deg = collections.Counter()
    for e in edges:
        deg[e[0]] += 1
        deg[e[1]] += 1
    return deg


def main():
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    df["route_type"] = df.route_type.str.strip().str.lower()
    df["A"] = df.from_town.map(canon)
    df["B"] = df.to_town.map(canon)

    # naive baseline (the trap)
    naive = collections.Counter(list(df.from_town) + list(df.to_town))
    print("Naive raw-string count leader:", naive.most_common(3))

    loops = df[df.A == df.B]
    edges = build(df)
    towns = set(df.A) | set(df.B)
    edge_towns = {x for e in edges for x in e}
    print(f"rows={len(df)}  self-loops dropped={len(loops)}")
    print(f"towns={len(towns)} (in edges: {len(edge_towns)})  connections={len(edges)}")
    print(f"connections if route_type distinguished = {len(build(df, True))}")

    deg = degrees(edges)
    ranked = sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))
    print("\nTop 10 by degree:")
    for i, (t, d) in enumerate(ranked[:10], 1):
        print(f"{i:2d}. {t.title():20s} {d}")
    top8 = [d for _, d in ranked[:8]]
    ok = len(towns) == 116 and len(edges) == 448 and len(set(top8)) == 8
    print("\nSelf-check (116 towns, 448 connections, top-8 distinct):", "PASS" if ok else "FAIL")
    flag = "SIGMOID_" + "".join(t[0].upper() for t, _ in ranked[:7])
    print("FLAG:", flag)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = ranked[:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f6feb" if (t, d) in ranked[:7] else "#9aa4b2" for t, d in top]
    ax.barh([t.title() for t, _ in top], [d for _, d in top], color=colors)
    for i, (_, d) in enumerate(top):
        ax.text(d + 0.2, i, str(d), va="center", fontsize=9)
    ax.set_xlabel("Distinct connections (degree)")
    ax.set_title(f"Top-15 towns, cleaned network ({len(towns)} towns, {len(edges)} connections)\n{flag}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "top15_degree.png", dpi=150)
    return df, edges, ranked


if __name__ == "__main__":
    main()
