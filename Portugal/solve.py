"""Sigmoid Portugal - The Douro Paradox.

Finds the single column whose split reverses the treatment-vs-control
harvest_success comparison in EVERY group (complete Simpson's paradox).

Usage:  python -I solve.py [path/to/douro_vineyard_data.csv]
Outputs: prints the analysis and the flag, writes douro_paradox.png next to this file.
"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "douro_vineyard_data.csv"
Y, T = "harvest_success", "treatment"
MIN_N = 30  # minimum plots per arm per group for a scanned split to count


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def diff_by_group(df, groups):
    """Return (treatment - control) success-rate difference per group, or None if a cell is too small."""
    g = df.groupby([groups, df[T]], observed=True)[Y].agg(["mean", "size"]).unstack()
    if g.shape[1] != 4 or g["size"].min().min() < MIN_N:
        return None
    return (g[("mean", "treatment")] - g[("mean", "control")]).values


def full_reversal(d):
    return d is not None and len(d) >= 2 and (d < 0).all()


def scan(df):
    """Search every column for any grouping that gives a complete reversal."""
    results = {}
    for col in df.columns:
        if col in ("plot_id", T, Y):
            continue
        hits, tried = [], 0
        x = df[col]
        if not pd.api.types.is_numeric_dtype(x):
            lv = sorted(x.unique())
            # natural levels
            tried += 1
            if full_reversal(diff_by_group(df, x)):
                hits.append(f"levels {lv}")
            # every 2-way partition of the levels
            for r in range(1, len(lv) // 2 + 1):
                for sub in itertools.combinations(lv, r):
                    tried += 1
                    if full_reversal(diff_by_group(df, x.isin(sub))):
                        hits.append(f"{set(sub)} vs rest")
        else:
            # quantile binnings
            for q in (2, 3, 4, 5, 10):
                tried += 1
                if full_reversal(diff_by_group(df, pd.qcut(x, q, duplicates="drop"))):
                    hits.append(f"{q}-quantile bins")
            # every single threshold at observed values (2 groups)
            vals = np.unique(x)
            for v in vals[1:]:
                tried += 1
                if full_reversal(diff_by_group(df, x >= v)):
                    hits.append(f"cut at {v}")
        results[col] = (tried, hits)
    return results


def main():
    df = pd.read_csv(DATA)
    print(f"Rows: {len(df)}")

    # ---- 1. overall rates -------------------------------------------------
    ov = df.groupby(T)[Y].agg(["sum", "size", "mean"])
    print("\nOverall harvest_success:")
    for arm, r in ov.iterrows():
        lo, hi = wilson(r["sum"], r["size"])
        print(f"  {arm:9s}: {int(r['sum']):4d}/{int(r['size']):4d} = {r['mean']:.2%}  (95% CI {lo:.1%}-{hi:.1%})")
    assert ov.loc["treatment", "mean"] > ov.loc["control", "mean"]

    # ---- 2. scan all columns --------------------------------------------
    print("\nScan of every column (categoricals: levels + all 2-way level partitions; "
          f"numerics: quantile bins + every threshold; min {MIN_N} plots per arm per group):")
    res = scan(df)
    for col, (tried, hits) in res.items():
        tag = "FULL REVERSAL" if hits else "no full reversal"
        extra = f"  [{hits[0]} ... {hits[-1]}; {len(hits)} splits]" if hits else ""
        print(f"  {col:24s} {tried:5d} groupings tried -> {tag}{extra}")
    winners = [c for c, (_, h) in res.items() if h]
    assert len(winners) == 1, winners
    col = winners[0]

    # ---- 3. the natural grouping: the gap in the distribution -----------
    vals = np.sort(df[col].unique())
    gaps = np.diff(vals)
    i = gaps.argmax()
    lo_edge, hi_edge = vals[i], vals[i + 1]
    print(f"\n{col}: largest gap in the distribution is {lo_edge} -> {hi_edge} km "
          f"(next largest gap {np.sort(gaps)[-2]:.1f}); no plot lies in between.")
    df["group"] = np.where(df[col] <= lo_edge, f"near (<= {lo_edge:g} km)", f"far (>= {hi_edge:g} km)")

    rows = []
    for grp in [f"near (<= {lo_edge:g} km)", f"far (>= {hi_edge:g} km)"]:
        sub = df[df.group == grp]
        for arm in ("control", "treatment"):
            s = sub[sub[T] == arm][Y]
            lo, hi = wilson(s.sum(), len(s))
            rows.append(dict(group=grp, arm=arm, k=int(s.sum()), n=len(s), rate=s.mean(), lo=lo, hi=hi))
    tab = pd.DataFrame(rows)
    print("\nPer-group harvest_success:")
    for grp, g in tab.groupby("group", sort=False):
        c, t = g[g.arm == "control"].iloc[0], g[g.arm == "treatment"].iloc[0]
        print(f"  {grp}: control {c.k}/{c.n} = {c.rate:.2%} | treatment {t.k}/{t.n} = {t.rate:.2%} "
              f"| diff {100*(t.rate-c.rate):+.2f} pp -> control ahead: {c.rate > t.rate}")
        share = t.n / (t.n + c.n)
        print(f"     share of plots on treatment in this group: {share:.1%}")

    # Mantel-Haenszel style stratified (group-size weighted) difference
    w = tab.groupby("group").n.sum()
    d = tab.pivot(index="group", columns="arm", values="rate")
    adj = ((d.treatment - d.control) * w).sum() / w.sum()
    print(f"\nGroup-size-weighted (standardised) difference: {100*adj:+.2f} pp "
          f"vs crude pooled difference {100*(ov.loc['treatment','mean']-ov.loc['control','mean']):+.2f} pp")

    # ---- 4. figure --------------------------------------------------------
    make_figure(ov, tab, col)

    flag = "SIGMOID_" + col.upper()
    print(f"\nFLAG: {flag}")
    return flag


def make_figure(ov, tab, col):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"control": "#4C72B0", "treatment": "#DD8452"}
    labels = ["Pooled (all plots)"] + list(tab.group.unique())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(labels))
    wdt = 0.36
    for j, arm in enumerate(("control", "treatment")):
        rates, err_lo, err_hi, ns = [], [], [], []
        r = ov.loc[arm]
        lo, hi = wilson(r["sum"], r["size"])
        rates.append(r["mean"]); err_lo.append(r["mean"] - lo); err_hi.append(hi - r["mean"]); ns.append(int(r["size"]))
        for grp in labels[1:]:
            g = tab[(tab.group == grp) & (tab.arm == arm)].iloc[0]
            rates.append(g.rate); err_lo.append(g.rate - g.lo); err_hi.append(g.hi - g.rate); ns.append(g.n)
        pos = x + (j - 0.5) * wdt
        ax.bar(pos, rates, wdt, color=colors[arm], label=arm, yerr=[err_lo, err_hi], capsize=4,
               error_kw=dict(ecolor="#555", lw=1))
        for p, rt, n in zip(pos, rates, ns):
            ax.text(p, 0.012, f"{rt:.1%}\nn={n}", ha="center", va="bottom", fontsize=8.5, color="white",
                    fontweight="bold")
    ax.axvline(0.5, color="#999", ls="--", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("harvest_success rate (95% Wilson CI)")
    ax.set_ylim(0, 0.45)
    ax.set_title(f"Douro paradox: treatment wins pooled, loses in every {col} group")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = HERE / "douro_paradox.png"
    fig.savefig(out, dpi=150)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
