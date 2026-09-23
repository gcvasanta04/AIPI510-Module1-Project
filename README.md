# The Only College in Town: Rural Education Deserts & For-Profit Monopoly Capture

**AIPI 510 — Module 1 Project**

## Overview

This project asks whether distance from a public college creates a smooth
decline in access, or something sharper. Using U.S. county-level data, we
find no clean "mile marker" where risk spikes. Instead, for-profit presence
in a rural county's local college market is nearly binary. About 2% of rural
counties with any local college market (29 of 1,397) have a market that is
**100% for-profit**, with no public alternative anywhere nearby. Most of
these are single-trade cosmetology or barber schools, not large debt-heavy
for-profit universities: debt and completion outcomes at these schools are
actually reasonable. The real harm isn't predatory debt; it's that residents
of these ~660,000-person counties have exactly one path into higher
education, regardless of what they're actually interested in.


## Datasets

| Source | What it provides | Link |
|---|---|---|
| College Scorecard (U.S. Dept. of Education) | Every U.S. postsecondary institution: location, ownership/control type, admission rate, enrollment, median debt, completion rate, median earnings | https://collegescorecard.ed.gov/data |
| American Community Survey, 5-year estimates (U.S. Census Bureau) | County population (B01003) and median household income (B19013), pulled via the Census API | https://api.census.gov/data.html |
| County Gazetteer Files (U.S. Census Bureau) | County centroid coordinates | https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html |
| Rural-Urban Continuum Codes (USDA Economic Research Service) | County-level rural/metro classification (RUCC 1–9) | https://www.ers.usda.gov/data-products/rural-urban-continuum-codes |

**Citation note:** College Scorecard and ACS data are public domain U.S.
government data; cite as "U.S. Department of Education, College Scorecard"
and "U.S. Census Bureau, American Community Survey 5-Year Estimates"
respectively. USDA RUCC data: "U.S. Department of Agriculture, Economic
Research Service, Rural-Urban Continuum Codes."

## Repository Structure

```
.
├── build_county_centroids.py       # builds county_centroids.csv from Gazetteer + ACS + RUCC
├── education_desert_boundary.py    # core analysis: distance, for-profit share, breakpoint tests
├── identify_monopoly_counties.py   # names the 100%-for-profit counties and pulls institution outcomes
├── data/
│   ├── raw/
│   │   ├── Most-Recent-Cohorts-Institution.csv   # NOT committed — download yourself, see below
│   │   ├── 2024_Gaz_counties_national.txt        # Census Gazetteer counties file (included)
│   │   └── Ruralurbancontinuumcodes2023.xlsx     # USDA RUCC file (included)
│   └── processed/
│       ├── county_centroids.csv
│       ├── county_desert_forprofit.csv
│       └── monopoly_counties.csv
└── README.md
```

The Gazetteer and RUCC files are small and included directly in `data/raw/`.
`Most-Recent-Cohorts-Institution.csv` is **not** committed (since its over the 100MB commit limit) so it must be downloaded separately
before running the pipeline (see Step 2 below). ACS population/income data
is pulled live from the Census API at run time rather than stored as a
static file.

## Reproducing the Analysis

### 1. Requirements

```
pip install pandas requests scikit-learn statsmodels piecewise-regression openpyxl python-dotenv
```

### 2. Download the College Scorecard raw data

Every other raw file needed is already included in `data/raw/`. The one
exception is the College Scorecard institution file (~100MB is too large to
commit directly):

1. Go to https://collegescorecard.ed.gov/data
2. Download **"Most Recent Institution-Level Data"** (labeled
   "Most-Recent-Cohorts-Institution" in the download list)
3. Unzip it and place `Most-Recent-Cohorts-Institution.csv` in `data/raw/`

### 3. Set up your Census API key

Get a free key: https://api.census.gov/data/key_signup.html

Create a `.env` file in the repo root (not committed — see `.gitignore`)
containing:

```
CENSUS_API_KEY=your_key_here
```

`build_county_centroids.py` loads this automatically via `python-dotenv`
(included in the requirements above).

### 4. Run the pipeline, in order

```bash
# Step 1: build the county-level covariate file (centroids, population,
# income, rural/metro code)
python build_county_centroids.py
# → outputs county_centroids.csv

# Step 2: compute distance-to-nearest-public-college and local for-profit
# market share for every county; run the breakpoint/robustness analysis
python education_desert_boundary.py
# → outputs county_desert_forprofit.csv, prints EDA diagnostics to console

# Step 3: identify the rural counties with a 100%-for-profit local market,
# and pull debt/completion/earnings outcomes for the institutions serving them
python identify_monopoly_counties.py
# → outputs monopoly_counties.csv
```
## Data Access

The Gazetteer and RUCC raw files are small and included directly in
`data/raw/`. `Most-Recent-Cohorts-Institution.csv` is not committed due to
its size (~100MB, right at GitHub's per-file limit) — see Step 2 above for
the exact download link and filename. ACS population/income data is pulled
live from the Census API each time the pipeline runs rather than stored as
a static file. The **processed/cleaned outputs** this project generated
(`county_centroids.csv`, `county_desert_forprofit.csv`,
`monopoly_counties.csv`) are committed under `data/processed/` and are what
the final analysis and visualizations are built from.

## Data Limitations & Ethical Considerations

- Institution enrollment is measured at the institution's location, not
  where students live. This describes the local *option mix*, not
  individual student decisions.
- "Local market" is defined by a 30-mile radius. The core finding (a
  near-binary rather than gradual pattern) held under radius sensitivity
  checks from 20–100 miles, though the exact county list shifts slightly.
- Race and gender were intentionally excluded from this analysis; this
  project examines one slice of educational-access inequality (geography
  and market structure), not the full picture.
- The 29 identified counties are the complete result of applying one fixed
  rule (for-profit share = 100%) to all 3,222 U.S. counties.