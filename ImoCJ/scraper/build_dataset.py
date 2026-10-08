"""
Turn the raw scraped text (data/raw/storia_listings_raw.csv) into:
  data/processed/listings_clean.csv - every cleaned column, NaN where unknown
  data/processed/listings_model.csv - model-ready: no data leaks, no metadata,
                                      only numeric features known for every row

  python scraper/build_dataset.py

What it does:
  * converts text codes to numbers (floor_3 -> 3, ground_floor -> 0, ...)
  * converts RON prices to EUR
  * fills missing build_year from the description when possible ("construit in 2019")
  * engineers features useful for regression: distance to the city centre,
    building age, top/ground floor flags, amenity 0/1 flags
  * removes obvious data-entry errors / outliers and duplicate re-posts
"""
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "storia_listings_raw.csv"
OUT_CSV = ROOT / "data" / "processed" / "listings_clean.csv"
MODEL_CSV = ROOT / "data" / "processed" / "listings_model.csv"

TARGET = "price_eur"
# data leaks (computed from the price) and metadata that says nothing about the flat
DROP_FOR_MODEL = ["id", "price_per_m2", "url", "created_at", "build_year_source", "n_images"]
# a feature is kept only if (almost) every listing has it; the few rows still
# missing it are dropped, so listings_model.csv has no NaN at all
MIN_COVERAGE = 0.99
# important features kept even though some listings lack them -
# listings without them are dropped from the model table
REQUIRED_FEATURES = ["floor", "build_year", "has_central_heating"]

RON_PER_EUR = 5.08          # approximate BNR rate, 2026 - few listings are in RON
CENTER = (46.7695, 23.5899)  # Piata Unirii, Cluj-Napoca
CURRENT_YEAR = 2026

# sanity limits for a monthly apartment rent in Cluj
PRICE_RANGE = (150, 6000)     # EUR / month
AREA_RANGE = (12, 300)        # m2
PPM2_RANGE = (3, 45)          # EUR / m2 / month


def floor_to_int(v):
    if pd.isna(v) or v == "":
        return pd.NA
    v = str(v)
    if v in ("ground_floor", "parter"):
        return 0
    if v in ("cellar", "basement", "semi_basement"):
        return -1
    if v == "floor_higher_10":
        return 11
    if v == "garret":  # mansarda - real floor unknown, handled via is_top_floor
        return pd.NA
    m = re.search(r"(\d+)", v)
    return int(m.group(1)) if m else pd.NA


YEAR_PATTERNS = [
    r"(?:construit|construc[tțţ]i[ea]|finalizat|recep[tțţ]ionat|dat [iî]n folosin[tțţ][aă])\D{0,25}((?:19|20)\d{2})",
    r"(?:bloc|imobil|cl[aă]dire|ansamblu)\s+(?:nou\s+)?(?:construit\s+)?(?:din|[iî]n)\s+(?:anul\s+)?((?:19|20)\d{2})",
    r"an(?:ul)?\s+(?:de\s+)?construc[tțţ]ie\D{0,10}((?:19|20)\d{2})",
]
NEW_BUILD_RE = re.compile(
    r"bloc nou|imobil nou|cl[aă]dire nou[aă]|construc[tțţ]ie nou[aă]|ansamblu nou|prima [iî]nchiriere|bloc nou-nou[tțţ]",
    re.I,
)


def year_from_text(text):
    if not isinstance(text, str):
        return pd.NA
    for pat in YEAR_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            y = int(m.group(1))
            if 1850 <= y <= CURRENT_YEAR + 2:
                return y
    return pd.NA


def haversine_km(lat, lon, lat0=CENTER[0], lon0=CENTER[1]):
    if pd.isna(lat) or pd.isna(lon):
        return pd.NA
    r = 6371.0
    p1, p2 = math.radians(lat0), math.radians(lat)
    dp, dl = p2 - p1, math.radians(lon - lon0)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def has_code(series, *codes):
    """1 if any of the codes appears in a pipe-separated list column."""
    codes = set(codes)
    return series.fillna("").apply(lambda s: int(bool(codes & set(s.split("|")))))


def has_text(series, pattern):
    return series.fillna("").str.contains(pattern, case=False, regex=True).astype(int)


def make_model_table(df):
    """Only leak-free numeric features that are known for (practically) every listing."""
    feats = df.drop(columns=DROP_FOR_MODEL + [TARGET]).select_dtypes("number")
    coverage = feats.notna().mean()
    keep = [c for c in feats.columns if coverage[c] >= MIN_COVERAGE or c in REQUIRED_FEATURES]
    dropped = coverage.drop(keep).sort_values()
    model = df[keep + [TARGET]].dropna().astype(float)
    print(f"\nmodel table: {len(keep)} features, {len(model)} of {len(df)} rows "
          f"(rows missing {', '.join(REQUIRED_FEATURES)} removed)")
    print("  kept   :", ", ".join(keep))
    print("  dropped (too many missing):",
          ", ".join(f"{c} {v:.0%}" for c, v in dropped.items()) or "-")
    print("  dropped (text/categorical):",
          ", ".join(df.drop(columns=DROP_FOR_MODEL).select_dtypes(exclude="number").columns))
    return model


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    raw = pd.read_csv(RAW_CSV, dtype=str)
    n0 = len(raw)
    print(f"raw listings: {n0}")
    raw = raw.drop_duplicates("id")

    df = pd.DataFrame({"id": raw["id"]})

    # --- target
    price = pd.to_numeric(raw["price"], errors="coerce")
    df["price_eur"] = price.where(raw["currency"].fillna("EUR") != "RON", price / RON_PER_EUR).round(0)

    # --- size
    df["area_m2"] = pd.to_numeric(raw["area_m2"], errors="coerce")
    df["rooms"] = pd.to_numeric(raw["rooms"].str.extract(r"(\d+)")[0], errors="coerce")
    df["price_per_m2"] = (df["price_eur"] / df["area_m2"]).round(2)

    # --- building
    df["floor"] = raw["floor"].apply(floor_to_int).astype("Int64")
    df["building_floors"] = pd.to_numeric(raw["building_floors"], errors="coerce").astype("Int64")
    df["is_ground_floor"] = (df["floor"] == 0).astype("Int64")
    is_garret = (raw["floor"] == "garret").to_numpy()
    top = ((df["floor"] >= df["building_floors"]) & (df["floor"] > 0)).astype("Int64")
    top[is_garret] = 1
    df["is_top_floor"] = top  # NA when floor or building height is unknown

    year = pd.to_numeric(raw["build_year"], errors="coerce")
    year = year.where(year.between(1800, CURRENT_YEAR + 3))
    text = raw["title"].fillna("") + " " + raw["description"].fillna("")
    year_txt = text.apply(year_from_text)
    df["build_year"] = year.fillna(year_txt).astype("Int64")
    df["build_year_source"] = "field"
    df.loc[year.isna() & year_txt.notna(), "build_year_source"] = "description"
    df.loc[df["build_year"].isna(), "build_year_source"] = "missing"
    df["building_age"] = (CURRENT_YEAR - df["build_year"]).clip(lower=0)
    df["is_new_building"] = ((df["build_year"] >= 2010).fillna(False)
                             | text.str.contains(NEW_BUILD_RE)).astype(int)

    df["building_type"] = raw["building_type"].fillna("unknown")
    df["building_material"] = raw["building_material"].fillna("unknown")
    df["heating"] = raw["heating"].fillna("unknown")
    df["windows_type"] = raw["windows_type"].fillna("unknown")

    # --- location
    df["neighborhood"] = raw["neighborhood"].fillna("unknown").replace("", "unknown")
    df["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    df["dist_center_km"] = [round(haversine_km(a, b), 3) if pd.notna(a) else pd.NA
                            for a, b in zip(df["latitude"], df["longitude"])]

    # --- amenities (0/1)
    ex, eq, sec, med, lift = raw["extras"], raw["equipment"], raw["security"], raw["media"], raw["lift"]
    df["has_lift"] = (has_code(ex, "lift") | (lift.fillna("") == "y").astype(int))
    df["has_balcony"] = has_code(ex, "balcony")
    df["has_terrace"] = has_code(ex, "terrace") | has_text(text, r"\bteras[aă]")
    df["has_parking"] = has_code(ex, "garage", "parking") | has_text(text, r"loc de parcare|parcare subteran|garaj")
    df["has_separate_kitchen"] = has_code(ex, "separate_kitchen")
    df["has_ac"] = has_code(ex, "air_conditioning") | has_code(eq, "air_conditioning") | has_text(text, r"aer condi[tțţ]ionat|\bclima\b")
    df["is_furnished"] = has_code(eq, "furniture") | has_text(text, r"\bmobilat")
    df["has_dishwasher"] = has_code(eq, "dishwasher") | has_text(text, r"ma[sșş]in[aă] de sp[aă]lat vase")
    df["has_washing_machine"] = has_code(eq, "washing_machine")
    df["has_internet"] = has_code(med, "internet")
    df["has_security"] = (sec.fillna("") != "").astype(int)
    df["pet_friendly"] = has_text(text, r"pet[ -]?friendly|accept[aă]m animale|animale (?:de companie )?(?:sunt )?acceptate")
    # own gas boiler ("centrala proprie"); NA when the heating type is unknown
    df["has_central_heating"] = (df["heating"] == "gas").astype("Int64").mask(df["heating"] == "unknown")

    # --- listing meta
    df["is_agency"] = (raw["advertiser_type"].fillna("") == "agency").astype(int)
    df["n_images"] = pd.to_numeric(raw["n_images"], errors="coerce")
    df["created_at"] = raw["created_at"]
    df["url"] = raw["url"]

    # --- cleaning
    before = len(df)
    df = df.dropna(subset=["price_eur", "area_m2"])
    df = df[df["price_eur"].between(*PRICE_RANGE)
            & df["area_m2"].between(*AREA_RANGE)
            & df["price_per_m2"].between(*PPM2_RANGE)]
    print(f"removed {before - len(df)} rows with missing/implausible price or area")

    # same apartment posted by several agencies: same price, size, floor and ~same place
    key = df.assign(lat_r=df["latitude"].round(3), lon_r=df["longitude"].round(3))
    dup = key.duplicated(subset=["price_eur", "area_m2", "rooms", "floor", "lat_r", "lon_r"])
    df = df[~dup]
    print(f"removed {int(dup.sum())} probable duplicate re-posts")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"clean listings: {len(df)} -> {OUT_CSV.relative_to(ROOT)}")
    print("\nmissing values (share):")
    print(df.isna().mean().round(3)[lambda s: s > 0].to_string())
    print("\nprice_eur summary:")
    print(df["price_eur"].describe().round(1).to_string())
    print("\nbuild_year source:", df["build_year_source"].value_counts().to_dict())

    model = make_model_table(df)
    model.to_csv(MODEL_CSV, index=False, encoding="utf-8")
    print(f"  -> {MODEL_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
