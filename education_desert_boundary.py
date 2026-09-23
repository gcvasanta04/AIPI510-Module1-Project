import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ----------------------------- CONFIG ------------------------------------
SCORECARD_CSV = "data/raw/Most-Recent-Cohorts-Institution.csv"
COUNTY_CSV    = "data/processed/county_centroids.csv"  # cols: fips, lat, lon, pop, median_income, rucc
BROAD_ACCESS_ADMIT_CUTOFF = 0.80        # admit rate >= this counts as broad-access
MARKET_RADIUS_MI = 30.0                 # radius defining a county's local "market"
EARTH_RADIUS_MI = 3958.8

# Known mega-online institutions whose enrollment is national, not local.
MEGA_ONLINE_UNITIDS = {
    484613,  # University of Phoenix
    183044,  # Southern New Hampshire University
    230603,  # Western Governors University
    433387,  # Grand Canyon University
    154022,  # Liberty University (large online)
}

# --------------------------- LOAD & CLASSIFY -----------------------------
def load_institutions(path):
    cols = ["UNITID", "INSTNM", "STABBR", "LATITUDE", "LONGITUDE",
            "CONTROL", "PREDDEG", "ICLEVEL", "ADM_RATE", "UGDS", "DISTANCEONLY"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)

    # Scorecard encodes suppressed/null as "PrivacySuppressed" or "NULL" — coerce.
    for c in ["LATITUDE", "LONGITUDE", "ADM_RATE", "UGDS", "DISTANCEONLY",
              "CONTROL", "PREDDEG", "ICLEVEL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    # Drop exclusively-online institutions and the known mega-online giants:
    # their enrollment is national, not local, and would poison geographic shares.
    df = df[df["DISTANCEONLY"] != 1]
    df = df[~df["UNITID"].isin(MEGA_ONLINE_UNITIDS)]

    # CRITICAL gotcha: ADM_RATE is NULL for open-admission schools (most community
    # colleges). Treat null admit rate as broad-access, NOT as missing-and-dropped.
    is_public     = df["CONTROL"] == 1
    is_forprofit  = df["CONTROL"] == 3
    broad_access  = (df["ADM_RATE"] >= BROAD_ACCESS_ADMIT_CUTOFF) | df["ADM_RATE"].isna()

    df["is_broad_public"] = is_public & broad_access
    df["is_forprofit"]    = is_forprofit
    df["ugds"]            = df["UGDS"].fillna(0.0)
    return df


# --------------------------- DISTANCE ------------------------------------
def nearest_public_distance_mi(county_ll, public_ll):
    """Great-circle distance from each county centroid to nearest broad-access public."""
    tree = BallTree(np.radians(public_ll), metric="haversine")
    dist_rad, _ = tree.query(np.radians(county_ll), k=1)
    return (dist_rad[:, 0] * EARTH_RADIUS_MI)


# ------------------- LOCAL FOR-PROFIT MARKET SHARE -----------------------
def forprofit_share_within_radius(counties, inst, radius_mi):
    """For each county centroid, for-profit share of enrollment within radius."""
    inst_ll   = np.radians(inst[["LATITUDE", "LONGITUDE"]].to_numpy())
    county_ll = np.radians(counties[["lat", "lon"]].to_numpy())
    tree = BallTree(inst_ll, metric="haversine")
    r = radius_mi / EARTH_RADIUS_MI
    idx_lists = tree.query_radius(county_ll, r=r)

    ugds = inst["ugds"].to_numpy()
    fp   = inst["is_forprofit"].to_numpy().astype(float)
    share = np.full(len(counties), np.nan)
    for i, idxs in enumerate(idx_lists):
        if len(idxs) == 0:
            continue
        total = ugds[idxs].sum()
        if total > 0:
            share[i] = (ugds[idxs] * fp[idxs]).sum() / total
    return share


def report_top_bin_detail(x, y, county_names=None, top_frac=0.1, label=""):
    """
    Shows what's actually driving the farthest-distance bin's mean — a handful
    of extreme outliers dragging the average look very different from a broad,
    population-level pattern. Print both to tell them apart.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) == 0:
        return
    cutoff = np.quantile(xm, 1 - top_frac)
    top_mask = xm >= cutoff
    top_y = ym[top_mask]
    top_x = xm[top_mask]

    print(f"\n  --- Top {top_frac:.0%} by distance ({label}): "
          f"n={top_mask.sum()}, distance >= {cutoff:.1f} mi ---")
    print(f"    mean share:   {top_y.mean():.3f}")
    print(f"    median share: {np.median(top_y):.3f}")
    print(f"    std share:    {top_y.std():.3f}")
    print(f"    max share:    {top_y.max():.3f}  (at {top_x[np.argmax(top_y)]:.1f} mi)")
    for thresh in (0.10, 0.25, 0.50):
        frac_above = (top_y >= thresh).mean()
        print(f"    share of these counties with fp_share >= {thresh:.0%}: {frac_above:.1%}")
    if (top_y.mean() - np.median(top_y)) > 0.03:
        print("    NOTE: mean is notably above median — a few high-share outliers "
              "are likely pulling the mean up more than a 'typical' far-away county shows.")


def radius_sensitivity(counties_base, inst, rucc_filter, radii, label=""):
    """
    Re-derives for-profit share (and the breakpoint) at several MARKET_RADIUS_MI
    values so we can see whether the ~30mi rural cliff is an artifact of one
    radius choice or holds up as the radius changes.
    """
    print(f"\n  === Radius sensitivity: {label} ===")
    for r in radii:
        sub = counties_base[rucc_filter].copy()
        sub["fp_share_r"] = forprofit_share_within_radius(sub, inst, r)
        x = sub["dist_public_mi"].to_numpy()
        y = sub["fp_share_r"].to_numpy()
        n_valid = np.isfinite(y).sum()
        n_zero_market = len(sub) - n_valid
        if n_valid < 20:
            print(f"    radius={r}mi: too few counties with a market (n={n_valid}).")
            continue
        bp, ci = find_breakpoint(x, y)
        print(f"    radius={r:>3}mi | n_with_market={n_valid:>4} | "
              f"n_zero_market={n_zero_market:>4} | breakpoint={bp:5.1f}mi",
              f"(95% CI {ci})" if ci else "(no CI — install piecewise-regression)")


def institutions_within_radius(counties, inst, radius_mi):
    """
    Count of institutions (any type) within radius of each county centroid,
    plus the for-profit count specifically. Lets us check whether a fp_share
    of exactly 1.0 really means 'one institution, and it's for-profit' (true
    monopoly capture) rather than something else.
    """
    inst_ll   = np.radians(inst[["LATITUDE", "LONGITUDE"]].to_numpy())
    county_ll = np.radians(counties[["lat", "lon"]].to_numpy())
    tree = BallTree(inst_ll, metric="haversine")
    r = radius_mi / EARTH_RADIUS_MI
    idx_lists = tree.query_radius(county_ll, r=r)

    fp = inst["is_forprofit"].to_numpy()
    n_total = np.array([len(idxs) for idxs in idx_lists])
    n_fp = np.array([fp[idxs].sum() if len(idxs) else 0 for idxs in idx_lists])
    return n_total, n_fp


def report_capture_vs_absence(counties, rucc_filter, label=""):
    """
    Histogram of fp_share to check bimodality (mostly 0%, a distinct cluster
    at/near 100%, little in between) vs. a smooth distribution. Also breaks
    down the fp_share==1.0 group by how many institutions are actually in
    their local market, to confirm 'monopoly capture' rather than an artifact.
    """
    sub = counties[rucc_filter]
    y = sub["fp_share"].to_numpy()
    valid = np.isfinite(y)
    y = y[valid]
    n_inst = sub["n_institutions"].to_numpy()[valid]

    print(f"\n  --- fp_share distribution ({label}), n={len(y)} ---")
    bins = [(-0.001, 0.001, "exactly 0%"),
            (0.001, 0.25, "0-25%"),
            (0.25, 0.50, "25-50%"),
            (0.50, 0.75, "50-75%"),
            (0.75, 0.999, "75-100% (not exact)"),
            (0.999, 1.001, "exactly 100%")]
    for lo, hi, name in bins:
        mask = (y > lo) & (y <= hi) if lo != -0.001 else (y <= 0.001)
        pct = mask.mean()
        print(f"    {name:>22}: {mask.sum():>4} counties ({pct:.1%})")

    captured = (y >= 0.999)
    if captured.sum() > 0:
        n_inst_captured = n_inst[captured]
        print(f"\n    Of the {captured.sum()} counties at ~100% for-profit share:")
        print(f"      exactly 1 institution in local market: "
              f"{(n_inst_captured == 1).mean():.1%}")
        print(f"      2-3 institutions (all for-profit): "
              f"{((n_inst_captured >= 2) & (n_inst_captured <= 3)).mean():.1%}")
        print(f"      4+ institutions (all for-profit): "
              f"{(n_inst_captured >= 4).mean():.1%}")
        print(f"      mean institutions in market: {n_inst_captured.mean():.1f}, "
              f"median: {np.median(n_inst_captured):.1f}")


# --------------------------- BREAKPOINT (segmented fit) ------------------------
def find_breakpoint(x, y):
    """
    Fit a 1-breakpoint segmented regression. Returns (breakpoint, ci) if the
    'piecewise-regression' package is available; else a grid-search fallback
    that reports the knot minimizing residual sum of squares.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    try:
        import piecewise_regression as pw
        fit = pw.Fit(x, y, n_breakpoints=1)
        res = fit.get_results()["estimates"]
        bp = res["breakpoint1"]["estimate"]
        ci = (res["breakpoint1"]["confidence_interval"])
        return bp, ci
    except Exception:
        # Fallback: grid search over candidate knots, min RSS piecewise-linear.
        cand = np.quantile(x, np.linspace(0.1, 0.9, 81))
        best_k, best_rss = None, np.inf
        for k in cand:
            X = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - k)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = ((y - X @ beta) ** 2).sum()
            if rss < best_rss:
                best_rss, best_k = rss, k
        return best_k, None


def crossing_distance(x, y, level=0.5):
    """Smallest distance at which a LOWESS-smoothed for-profit share crosses `level`."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    mask = np.isfinite(x) & np.isfinite(y)
    sm = lowess(y[mask], x[mask], frac=0.3, return_sorted=True)
    above = sm[sm[:, 1] >= level]
    return None if len(above) == 0 else float(above[0, 0])


# --------------------------- DIAGNOSTICS ------------------------------------
def print_decile_table(x, y, label=""):
    """
    Bins distance into deciles and prints mean for-profit share per bin.
    This is the single fastest way to SEE the shape of the relationship —
    a breakpoint statistic can hide or overstate what's actually there.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) == 0:
        print(f"  [{label}] no data to bin.")
        return
    df = pd.DataFrame({"dist": xm, "share": ym})
    try:
        df["bin"] = pd.qcut(df["dist"], 10, duplicates="drop")
    except ValueError:
        print(f"  [{label}] not enough distinct distances to form deciles.")
        return
    table = df.groupby("bin", observed=True).agg(
        n=("share", "size"),
        dist_min=("dist", "min"),
        dist_max=("dist", "max"),
        mean_fp_share=("share", "mean"),
    )
    print(f"\n  --- Distance deciles ({label}) ---")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))


def report_zero_market_counties(counties):
    """
    Counties with no institution within MARKET_RADIUS_MI have an undefined
    for-profit share (NaN) and are silently dropped from the regression.
    These are arguably the MOST extreme deserts — worth reporting separately,
    not just excluding.
    """
    zero_market = counties[counties["fp_share"].isna()]
    print(f"\n  Counties with NO institution within {MARKET_RADIUS_MI} mi "
          f"(excluded from the curve): {len(zero_market)} of {len(counties)}")
    if len(zero_market) > 0:
        print(f"    Their distance-to-nearest-public stats: "
              f"min={zero_market['dist_public_mi'].min():.1f}, "
              f"median={zero_market['dist_public_mi'].median():.1f}, "
              f"max={zero_market['dist_public_mi'].max():.1f}")
        if "rucc" in zero_market.columns:
            rural = zero_market["rucc"].fillna(-1) >= 4
            print(f"    Rural (RUCC>=4) share among these: "
                  f"{rural.mean():.1%}" if len(zero_market) else "n/a")


# ------------------------------- MAIN ------------------------------------
def main():
    inst = load_institutions(SCORECARD_CSV)
    counties = pd.read_csv(COUNTY_CSV)  # fips, lat, lon, pop, median_income, rucc

    public_ll = inst.loc[inst["is_broad_public"], ["LATITUDE", "LONGITUDE"]].to_numpy()
    county_ll = counties[["lat", "lon"]].to_numpy()

    counties["dist_public_mi"] = nearest_public_distance_mi(county_ll, public_ll)
    counties["fp_share"] = forprofit_share_within_radius(counties, inst, MARKET_RADIUS_MI)
    counties["n_institutions"], counties["n_forprofit"] = institutions_within_radius(
        counties, inst, MARKET_RADIUS_MI)

    x = counties["dist_public_mi"].to_numpy()
    y = counties["fp_share"].to_numpy()

    bp, ci = find_breakpoint(x, y)
    cross = crossing_distance(x, y, level=0.5)

    print(f"n counties with a local market: {np.isfinite(y).sum()}")
    print(f"segmented-regression breakpoint: {bp:.1f} mi", f"(95% CI {ci})" if ci else "(grid-search, no CI)")
    print(f"distance where for-profit share crosses 50%: {cross}")

    # --- Diagnostics: SEE the shape before trusting the single breakpoint stat ---
    print_decile_table(x, y, label="ALL counties (pooled)")
    report_zero_market_counties(counties)

    # --- Rural vs. metro split ---
    if "rucc" in counties.columns and counties["rucc"].notna().any():
        metro = counties["rucc"] <= 3
        rural = counties["rucc"] >= 4

        for label, mask_group in [("METRO (RUCC 1-3)", metro), ("RURAL (RUCC 4-9)", rural)]:
            sub = counties[mask_group]
            xs, ys = sub["dist_public_mi"].to_numpy(), sub["fp_share"].to_numpy()
            n_valid = np.isfinite(ys).sum()
            print(f"\n=== {label}: n counties={len(sub)}, n with local market={n_valid} ===")
            if n_valid < 20:
                print("  Too few counties with a computed share to fit a breakpoint.")
                continue
            bp_g, ci_g = find_breakpoint(xs, ys)
            cross_g = crossing_distance(xs, ys, level=0.5)
            print(f"  breakpoint: {bp_g:.1f} mi", f"(95% CI {ci_g})" if ci_g else "(grid-search, no CI)")
            print(f"  distance where for-profit share crosses 50%: {cross_g}")
            print_decile_table(xs, ys, label=label)
            report_top_bin_detail(xs, ys, top_frac=0.1, label=label)
    else:
        print("\n  No RUCC data available — skipping rural/metro split. "
              "Check that county_centroids.csv has a populated 'rucc' column.")

    # --- Radius sensitivity check (rural only — this is where the signal is) ---
    if "rucc" in counties.columns and counties["rucc"].notna().any():
        rural_filter = counties["rucc"] >= 4
        metro_filter = counties["rucc"] <= 3
        radius_sensitivity(counties, inst, rural_filter,
                            radii=[20, 30, 50, 75, 100], label="RURAL (RUCC 4-9)")

        report_capture_vs_absence(counties, rural_filter, label="RURAL (RUCC 4-9)")
        report_capture_vs_absence(counties, metro_filter, label="METRO (RUCC 1-3)")

    counties.to_csv("county_desert_forprofit.csv", index=False)


if __name__ == "__main__":
    main()