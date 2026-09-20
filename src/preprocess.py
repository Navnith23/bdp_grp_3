"""
Preprocessing for the keralarealestate.com listings (final.json) -> model-ready table.

Run:  python preprocess.py final.json cleaned.csv

Stages (each is its own function so we can tweak one at a time):
  1. load_and_flatten          - JSON dict-of-dicts -> flat DataFrame
  2. parse_fields              - text -> numbers (price, areas, age, beds, baths, date, district, type)
  3. clean_rows                - drop rent-only listings, rows with no/invalid target, duplicate URLs/titles
  4. validate_values           - impossible feature values -> NaN
  5. drop_implausible_prices   - domain-bound check of price per cent / per sq ft (fixed bounds, no learning)
  6. drop_duplicate_listings   - drop listings identical on every model-relevant field
  7. cap_outliers              - winsorise upper tail of numeric features (price is NOT capped)
  8. impute_missing            - grouped-median imputation (+ missing flags)
  9. add_features              - amenity flags, locality frequency, log target, one-hot (drop_first)

Steps 1-6 are data-quality corrections with fixed rules, so they are safe to run BEFORE a
train/test split. Steps 7-9 learn statistics (caps, medians, frequencies) - when a split
is introduced, those must be fitted on the training set only.
"""
import json
import re
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
MIN_VALID_PRICE = 100_000        # below this a "sale" price is almost surely per-cent/rent/typo
CAP_UPPER = 0.99                 # upper winsorise quantile for numeric features
AMENITY_MIN_SHARE = 0.03         # keep amenity flags present in >=3% of listings
# Domain-based plausibility bounds (Rs). Fixed rules, NOT learned from the data.
LAND_PRICE_PER_CENT = (2_000, 10_000_000)       # Rs 2 lakh/acre .. Rs 1 crore/cent
# Price per sq ft of BUILT-UP area. Houses/villas include the land in the price, so a high
# figure is often legitimate (large estates) -> only a lower bound is applied to them.
BUILDING_PRICE_PER_SQFT = {
    "apartment": (750, 40_000),
    "house": (750, None),
    "villa": (750, None),
}
MIN_GROUP_SIZE = 5               # min observed values for a district x type median to be trusted
IMPUTE_COLS = ["land_area_cents", "property_age", "built_up_sqft"]

# hard validity ranges: outside -> treated as invalid (set to NaN)
VALID_RANGES = {
    "land_area_cents": (0.5, 5_000),      # 5,000 cents = 50 acres
    "built_up_sqft": (100, 50_000),
    "bedrooms": (0, 20),
    "bathrooms": (0, 20),
    "property_age": (0, 150),
    "parking": (0, 50),
}


# ----------------------------------------------------------------- 1. load
def load_and_flatten(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for url, v in raw.items():
        info = v.get("listingInfo") or {}
        basic = v.get("basicInfo") or {}

        def first(*keys):
            """First non-empty value among listingInfo keys."""
            for k in keys:
                if info.get(k) not in (None, ""):
                    return info[k]
            return None

        rows.append(
            {
                "url": url,
                "title": v.get("listingTitle"),
                "price": v.get("listingPrice"),
                "listing_id": v.get("listingID"),
                "place": v.get("listingPlace"),
                "description": v.get("descriptionText") or "",
                "amenities": v.get("amenities") or [],
                # basicInfo is the cleanest source, listingInfo is the fallback
                "land_raw": basic.get("land_area") or first("land_area", "total_land_area", "plot_area"),
                "built_raw": basic.get("buildup_area") or first("built_up_area", "total_area"),
                "beds_raw": basic.get("beds") or first("bedrooms"),
                "baths_raw": basic.get("baths") or first("bathrooms"),
                "age_raw": basic.get("property_age") or first("property_age", "age_of_property"),
                "parking_raw": basic.get("parking"),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 2. parse
_NUM = r"(\d[\d,]*\.?\d*)"


def _first_number(s):
    m = re.search(_NUM, str(s))
    return float(m.group(1).replace(",", "")) if m else np.nan


def parse_land_cents(s):
    """'7.5 cents' / '2.5 acres' / '182 cent' -> cents (1 acre = 100 cents)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).lower()
    n = _first_number(s)
    if np.isnan(n):
        return np.nan
    if "acre" in s:
        return n * 100
    if "sq" in s:                       # sq ft given for land
        return n / 435.6
    return n                            # cents (also bare numbers)


def parse_sqft(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).lower()
    n = _first_number(s)
    if np.isnan(n):
        return np.nan
    if "cent" in s:                     # a few land areas typed into the built-up field
        return np.nan
    if re.search(r"sq\.?\s*(m|mtr|meter)", s):
        return n * 10.764
    return n


def parse_int(s):
    n = _first_number(s)
    return n if not np.isnan(n) else np.nan


def parse_age(raw, listing_year):
    """basicInfo gives a construction YEAR; listingInfo gives '10 years'. -> age in years."""
    if raw is None:
        return np.nan
    s = str(raw).lower()
    if re.search(r"\bnew\b|brand", s):
        return 0.0
    n = _first_number(s)
    if np.isnan(n):
        return np.nan
    if 1900 <= n <= 2100:               # a year
        return float(listing_year - n) if pd.notna(listing_year) else np.nan
    return n


def classify_type(title: str) -> str:
    t = title.lower() if isinstance(title, str) else ""
    if re.search(r"apartment|flat", t):
        return "apartment"
    if "villa" in t:
        return "villa"
    if re.search(r"house|home|bhk|bungalow", t):
        return "house"
    if re.search(r"land|plot|acre|cent", t):
        return "land"
    if re.search(r"commercial|shop|office|building|factory|quarry|petrol|hotel|resort|showroom|warehouse|mall|space", t):
        return "commercial"
    return "other"


def parse_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # listing_id looks like 'ERR101044 | 19 Sep 2026'
    df["listing_date"] = pd.to_datetime(
        df["listing_id"].str.split("|").str[-1].str.strip(), format="%d %b %Y", errors="coerce"
    )
    df["listing_year"] = df["listing_date"].dt.year

    parts = df["place"].fillna("").str.split("/")
    df["locality"] = parts.str[0].str.strip().str.lower().replace("", np.nan)
    df["district"] = (
        parts.str[-1].str.replace(r"\(.*?\)", "", regex=True).str.strip().str.title().replace("", np.nan)
    )

    df["property_type"] = df["title"].map(classify_type)
    t_low = df["title"].fillna("").str.lower()
    df["is_rent_only"] = t_low.str.contains(r"\brent|\blease", regex=True) & ~t_low.str.contains("sale")

    df["land_area_cents"] = df["land_raw"].map(parse_land_cents)
    df["built_up_sqft"] = df["built_raw"].map(parse_sqft)
    df["bedrooms"] = df["beds_raw"].map(parse_int)
    df["bathrooms"] = df["baths_raw"].map(parse_int)
    df["parking"] = df["parking_raw"].map(parse_int)
    df["property_age"] = [parse_age(a, y) for a, y in zip(df["age_raw"], df["listing_year"])]
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


# ----------------------------------------------------------------- 3. rows
def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[~df["is_rent_only"]]
    print(f"  dropped {n0 - len(df):>5} rent-only listings")

    n = len(df)
    df = df[df["price"].notna()]
    print(f"  dropped {n - len(df):>5} rows with no price (target missing)")

    n = len(df)
    df = df[df["price"] >= MIN_VALID_PRICE]
    print(f"  dropped {n - len(df):>5} rows with price < {MIN_VALID_PRICE:,} (invalid)")

    n = len(df)
    df = df.drop_duplicates(subset="url").drop_duplicates(subset=["title", "price", "place"])
    print(f"  dropped {n - len(df):>5} duplicates")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------- 4. validate
def validate_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in VALID_RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        if bad.any():
            print(f"  {col}: {bad.sum()} invalid values -> NaN (valid range {lo}-{hi})")
        df.loc[bad, col] = np.nan
    # bathrooms above bedrooms+3 is almost always a typo
    return df


# ----------------------------------------------------------------- 5. plausibility
def drop_implausible_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose price is impossible given their own area (data-entry errors such as a
    per-cent price typed as the total, or an acre value typed as cents).
    Uses OBSERVED values only, so it runs before imputation. Plausible expensive houses /
    large plots are kept - the bounds are domain limits, not statistical outlier cuts."""
    df = df.copy()
    bad = pd.Series(False, index=df.index)

    lo, hi = LAND_PRICE_PER_CENT
    land = df["property_type"].eq("land") & df["land_area_cents"].notna()
    ppc = df["price"] / df["land_area_cents"]
    bad_lo, bad_hi = land & (ppc < lo), land & (ppc > hi)
    print(f"  land: {bad_lo.sum()} rows below Rs {lo:,}/cent, {bad_hi.sum()} above Rs {hi:,}/cent")
    bad |= bad_lo | bad_hi

    for ptype, (lo, hi) in BUILDING_PRICE_PER_SQFT.items():
        m = df["property_type"].eq(ptype) & df["built_up_sqft"].notna()
        pps = df["price"] / df["built_up_sqft"]
        b_lo = m & (pps < lo)
        b_hi = m & (pps > hi) if hi is not None else pd.Series(False, index=df.index)
        print(f"  {ptype}: {b_lo.sum()} rows below Rs {lo:,}/sqft, {b_hi.sum()} above "
              + (f"Rs {hi:,}/sqft" if hi else "(no upper bound)"))
        bad |= b_lo | b_hi

    print(f"  dropped {bad.sum():>5} implausible-price rows")
    return df[~bad].reset_index(drop=True)


# ----------------------------------------------------------------- 6. duplicates
DEDUP_COLS = ["price", "property_type", "district", "locality", "land_area_cents", "built_up_sqft",
              "bedrooms", "bathrooms", "property_age", "parking", "listing_year"]


def drop_duplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Same property re-posted: identical on every model-relevant field (incl. amenities).
    Must happen before any train/test split so copies can't land on both sides."""
    key = df[DEDUP_COLS].copy()
    key["amenities"] = df["amenities"].map(lambda L: tuple(sorted({x.strip().lower() for x in L})))
    dup = key.duplicated(keep="first")
    print(f"  dropped {dup.sum():>5} duplicate listings")
    return df[~dup].reset_index(drop=True)


# ----------------------------------------------------------------- 7. cap (before imputing)
def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise the UPPER tail only (counts like 1 bedroom are legitimate, so no floor).
    Computed on observed values before imputation. Price is intentionally left uncapped."""
    df = df.copy()
    for col in ["land_area_cents", "built_up_sqft", "bedrooms", "bathrooms", "property_age", "parking"]:
        hi = df[col].quantile(CAP_UPPER)
        df[f"{col}_capped"] = (df[col] > hi).astype(int)
        df[col] = df[col].clip(upper=hi)
    return df


# ----------------------------------------------------------------- 8. impute
def _grouped_median(df, col):
    """District x type median (only where >= MIN_GROUP_SIZE observed) -> type median -> global."""
    obs = df[col].notna()
    cnt = obs.groupby([df["district"], df["property_type"]]).transform("sum")
    g_dt = df.groupby(["district", "property_type"])[col].transform("median").where(cnt >= MIN_GROUP_SIZE)
    g_t = df.groupby("property_type")[col].transform("median")
    return df[col].fillna(g_dt).fillna(g_t).fillna(df[col].median())


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Grouped-median imputation with missing flags.
    Structural zeros (not 'missing'): plots have no built-up area/beds/baths/age,
    apartments have no land area of their own."""
    df = df.copy()
    is_land = df["property_type"].eq("land")
    is_apt = df["property_type"].eq("apartment")

    for col in ["built_up_sqft", "bedrooms", "bathrooms", "property_age"]:
        df.loc[is_land, col] = df.loc[is_land, col].fillna(0)
    df.loc[is_apt, "land_area_cents"] = 0.0

    # bedrooms/bathrooms come from the same field and are ~always missing together, so
    # one flag (bedrooms_missing) covers both.
    for col in IMPUTE_COLS + ["bedrooms", "bathrooms"]:
        if col != "bathrooms":
            df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = _grouped_median(df, col)

    df["parking"] = df["parking"].fillna(0)   # not listed = no parking mentioned
    return df


# ----------------------------------------------------------------- 9. features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_price"] = np.log1p(df["price"])
    df["listing_year"] = df["listing_year"].astype(int)

    # de-duplicate each listing's amenities, then multi-hot the common ones
    sets = df["amenities"].map(lambda L: {x.strip().lower() for x in L})
    df["n_amenities"] = sets.map(len)
    counts = sets.explode().dropna().value_counts() / len(df)
    for a in counts[counts >= AMENITY_MIN_SHARE].index:
        df["amen_" + re.sub(r"\W+", "_", a).strip("_")] = sets.map(lambda S, a=a: int(a in S))

    # locality: too many values to one-hot -> frequency encoding (missing -> 0)
    freq = df["locality"].map(df["locality"].value_counts(normalize=True))
    df["locality_freq"] = freq.fillna(0)

    df = pd.get_dummies(df, columns=["district", "property_type"], prefix=["dist", "type"],
                        drop_first=True, dtype=int)
    return df.drop(columns=["amenities", "description", "locality"])


# ----------------------------------------------------------------- main
def main(src="final.json", dst="cleaned.csv"):
    print("1. load");         df = load_and_flatten(src);  print(f"  {len(df)} listings")
    print("2. parse");        df = parse_fields(df)
    print("3. rows");         df = clean_rows(df)
    print("4. validate");     df = validate_values(df)
    print("5. plausibility"); df = drop_implausible_prices(df)
    print("6. duplicates");   df = drop_duplicate_listings(df)
    print("7. outliers");     df = cap_outliers(df)
    print("8. impute");       df = impute_missing(df)
    print("9. features");     df = add_features(df)

    drop_cols = ["url", "title", "listing_id", "place", "land_raw", "built_raw", "beds_raw",
                 "baths_raw", "age_raw", "parking_raw", "is_rent_only", "listing_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # safety net: distinct listings can become identical after imputation/encoding
    dup = df.duplicated()
    print(f"  dropped {dup.sum():>5} rows identical after encoding")
    df = df[~dup].reset_index(drop=True)
    df.to_csv(dst, index=False)
    print(f"done: {df.shape[0]} rows x {df.shape[1]} cols -> {dst}")
    return df


if __name__ == "__main__":
    main(*(sys.argv[1:3]))