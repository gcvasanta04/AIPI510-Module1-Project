import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ----------------------------- CONFIG ------------------------------------
COUNTY_RESULTS_CSV = "data/processed/county_desert_forprofit.csv"      # output of the prior script
SCORECARD_CSV       = "data/raw/Most-Recent-Cohorts-Institution.csv"
OUTPUT_CSV           = "data/processed/monopoly_counties.csv"

# Must match the MARKET_RADIUS_MI used in education_desert_boundary.py, since
# fp_share there was computed at that radius — we need the same radius to
# find the same institutions that produced each county's fp_share.
MARKET_RADIUS_MI = 30.0
EARTH_RADIUS_MI = 3958.8

FP_SHARE_CUTOFF = 0.999   # "exactly 100%" for-profit, matching the prior report
RUCC_RURAL_CUTOFF = 4     # RUCC >= 4 counts as rural/nonmetro

# Same mega-online exclusion as the main script — keep in sync.
MEGA_ONLINE_UNITIDS = {
    484613, 183044, 230603, 433387, 154022,
}

# Scorecard outcome fields we want for the captor institutions.
# (Names can shift slightly by vintage — the script checks what's available
# and tells you if one is missing rather than failing silently.)
OUTCOME_FIELDS = {
    "UNITID": "UNITID",
    "INSTNM": "name",
    "STABBR": "state",
    "CONTROL": "control",
    "UGDS": "undergrad_enrollment",
    "ADM_RATE": "admit_rate",
    "DEBT_MDN": "median_debt",             # median federal debt at graduation
    "C150_4": "completion_rate_4yr",       # 150%-time completion, 4-yr
    "C150_L4": "completion_rate_2yr",      # 150%-time completion, <4-yr
    "MD_EARN_WNE_P10": "median_earnings_10yr",
}


def load_scorecard_outcomes(path):
    wanted = list(OUTCOME_FIELDS.keys())
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f"  Note: these Scorecard fields weren't found and will be skipped: {missing}")
    for c in df.columns:
        if c not in ("INSTNM", "STABBR"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[~df["UNITID"].isin(MEGA_ONLINE_UNITIDS)]
    return df.rename(columns={k: v for k, v in OUTCOME_FIELDS.items() if k in df.columns})


def find_institutions_near(county_row, inst_ll_rad, inst_df, radius_mi):
    """All institutions within radius_mi of one county's centroid."""
    r = radius_mi / EARTH_RADIUS_MI
    county_ll = np.radians([[county_row["lat"], county_row["lon"]]])
    tree_idx = BallTree(inst_ll_rad, metric="haversine")
    idxs = tree_idx.query_radius(county_ll, r=r)[0]
    return inst_df.iloc[idxs]


def main():
    print("Loading county results...")
    counties = pd.read_csv(COUNTY_RESULTS_CSV)

    print("Loading Scorecard outcome fields...")
    inst = load_scorecard_outcomes(SCORECARD_CSV)
    # Need lat/lon for the radius lookup too — reload minimally for that.
    latlon = pd.read_csv(SCORECARD_CSV, usecols=["UNITID", "LATITUDE", "LONGITUDE"],
                          low_memory=False)
    latlon["LATITUDE"] = pd.to_numeric(latlon["LATITUDE"], errors="coerce")
    latlon["LONGITUDE"] = pd.to_numeric(latlon["LONGITUDE"], errors="coerce")
    inst = inst.merge(latlon, on="UNITID", how="left").dropna(subset=["LATITUDE", "LONGITUDE"])
    inst = inst[~inst["UNITID"].isin(MEGA_ONLINE_UNITIDS)]
    inst_ll_rad = np.radians(inst[["LATITUDE", "LONGITUDE"]].to_numpy())

    # --- Filter to rural monopoly counties ---
    monopoly = counties[
        (counties["rucc"] >= RUCC_RURAL_CUTOFF) &
        (counties["fp_share"] >= FP_SHARE_CUTOFF)
    ].copy()
    print(f"\nFound {len(monopoly)} rural counties with ~100% for-profit local market.")

    if len(monopoly) == 0:
        print("Nothing to report — check FP_SHARE_CUTOFF / RUCC_RURAL_CUTOFF "
              "or that county_desert_forprofit.csv has the expected columns.")
        return

    total_pop = monopoly["pop"].sum(skipna=True)
    print(f"Total population living in these counties: {total_pop:,.0f}")
    print(f"Median county population: {monopoly['pop'].median():,.0f}")
    print(f"Median distance to nearest public college: "
          f"{monopoly['dist_public_mi'].median():.1f} mi")
    if "median_income" in monopoly.columns:
        print(f"Median household income in these counties: "
              f"${monopoly['median_income'].median():,.0f}")

    # --- For each monopoly county, find its actual captor institution(s) ---
    rows = []
    for _, county in monopoly.iterrows():
        nearby = find_institutions_near(county, inst_ll_rad, inst, MARKET_RADIUS_MI)
        for _, institution in nearby.iterrows():
            rows.append({
                "county_fips": county.get("fips"),
                "county_name": county.get("county_name"),
                "state": county.get("state"),
                "county_pop": county.get("pop"),
                "county_median_income": county.get("median_income"),
                "dist_to_nearest_public_mi": county.get("dist_public_mi"),
                **{k: institution.get(k) for k in
                   ["UNITID", "name", "control", "undergrad_enrollment", "admit_rate",
                    "median_debt", "completion_rate_4yr", "completion_rate_2yr",
                    "median_earnings_10yr"] if k in institution.index},
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(result)} county-institution rows to {OUTPUT_CSV}")

    print("\n--- Captor institution outcomes (rural monopoly counties) ---")
    for col, desc in [("median_debt", "Median federal debt at graduation ($)"),
                       ("completion_rate_4yr", "150%-time completion rate, 4-yr (0-1)"),
                       ("completion_rate_2yr", "150%-time completion rate, <4-yr (0-1)"),
                       ("median_earnings_10yr", "Median earnings 10yr after entry ($)")]:
        if col in result.columns and result[col].notna().any():
            print(f"  {desc}: median={result[col].median():,.1f}, "
                  f"mean={result[col].mean():,.1f}, n={result[col].notna().sum()}")
        else:
            print(f"  {desc}: not available in this Scorecard file/vintage.")

if __name__ == "__main__":
    main()