import io
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()
# ----------------------------- CONFIG -------------------------------------
GAZETTEER_FILE  = "data/raw/2026_Gaz_counties_national.txt"   # tab-delimited, from Census
RUCC_FILE       = "data/raw/Ruralurbancontinuumcodes2023.xlsx"  # from USDA ERS
CENSUS_API_KEY  = os.getenv("CENSUS_API_KEY", "")          # optional but recommended; "" works at low volume
ACS_YEAR        = 2022        # latest year with full 5-year estimates at time of writing
OUTPUT_FILE     = "data/processed/county_centroids.csv"


# --------------------------- 1. GAZETTEER ----------------------------------
def load_gazetteer(path):
    """
    Census Gazetteer county files have fixed columns including GEOID (5-digit
    county FIPS), NAME, USPS (state abbrev), INTPTLAT, INTPTLONG. The delimiter
    has changed across years (older files: tab; newer files, e.g. 2026: pipe "|"),
    so detect it from the header line instead of assuming one.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline()
    if header.count("|") > header.count("\t"):
        sep = "|"
    else:
        sep = "\t"

    df = pd.read_csv(path, sep=sep, dtype=str, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    def find_col(prefix):
        matches = [c for c in df.columns if c.upper().startswith(prefix)]
        if not matches:
            raise KeyError(f"No column starting with {prefix!r} in {path}. "
                            f"Available columns: {list(df.columns)}")
        return matches[0]

    geoid_col = find_col("GEOID")
    name_col  = find_col("NAME")
    usps_col  = find_col("USPS")
    lat_col   = find_col("INTPTLAT")
    lon_col   = find_col("INTPTLONG")

    out = df[[geoid_col, name_col, usps_col, lat_col, lon_col]].copy()
    out.columns = ["fips", "county_name", "state", "lat", "lon"]
    out["fips"] = out["fips"].str.strip().str.zfill(5)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    return out


# --------------------------- 2. ACS (population, income) -------------------
def fetch_acs(year, api_key, max_retries=3):
    """
    Pulls county-level population (B01003_001E) and median household income
    (B19013_001E) for all counties in one call via the Census API.
    """
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B01003_001E,B19013_001E",
        "for": "county:*",
        "in": "state:*",
    }
    if api_key:
        params["key"] = api_key

    for attempt in range(max_retries):
        resp = requests.get(base, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
            df = df.rename(columns={"B01003_001E": "pop", "B19013_001E": "median_income"})
            df["pop"] = pd.to_numeric(df["pop"], errors="coerce")
            # ACS codes suppressed/unavailable income as negative sentinel values.
            df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")
            df.loc[df["median_income"] < 0, "median_income"] = pd.NA
            return df[["fips", "pop", "median_income"]]
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Census API request failed after {max_retries} attempts: "
                        f"{resp.status_code} {resp.text[:200]}")


# --------------------------- 3. USDA RUCC -----------------------------------
def load_rucc(path):
    """
    USDA RUCC file has a FIPS column (5-digit county code) and an RUCC_2023
    (or similarly named) column, 1-9 where 1-3 = metro, 4-9 = nonmetro/rural.
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    fips_col = next((c for c in df.columns if c.upper() == "FIPS"), None)
    rucc_col = next((c for c in df.columns if "RUCC" in c.upper()), None)
    if fips_col is None or rucc_col is None:
        raise KeyError(f"Couldn't find FIPS/RUCC columns in {path}. "
                        f"Available columns: {list(df.columns)}")

    out = df[[fips_col, rucc_col]].copy()
    out.columns = ["fips", "rucc"]
    out["fips"] = out["fips"].str.strip().str.zfill(5)
    out["rucc"] = pd.to_numeric(out["rucc"], errors="coerce")
    return out


# --------------------------------- MAIN -------------------------------------
def main():
    print("Loading Gazetteer county centroids...")
    gaz = load_gazetteer(GAZETTEER_FILE)
    print(f"  {len(gaz)} counties")

    print("Fetching ACS population + median income from the Census API...")
    acs = fetch_acs(ACS_YEAR, CENSUS_API_KEY)
    print(f"  {len(acs)} counties")

    print("Loading USDA Rural-Urban Continuum Codes...")
    rucc = load_rucc(RUCC_FILE)
    print(f"  {len(rucc)} counties")

    merged = gaz.merge(acs, on="fips", how="left").merge(rucc, on="fips", how="left")

    missing_pop  = merged["pop"].isna().sum()
    missing_rucc = merged["rucc"].isna().sum()
    if missing_pop or missing_rucc:
        print(f"  Warning: {missing_pop} counties missing ACS data, "
              f"{missing_rucc} missing RUCC (often PR/territories or code mismatches — "
              f"check FIPS formatting if this number is large).")

    merged = merged[["fips", "county_name", "state", "lat", "lon",
                      "pop", "median_income", "rucc"]]
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE} with {len(merged)} counties.")


if __name__ == "__main__":
    main()