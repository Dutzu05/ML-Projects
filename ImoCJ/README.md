# ImoCJ: apartment rent prediction for Cluj-Napoca

A learning project that goes with a **System Identification** course. The goal is to predict the monthly rent (EUR) of an apartment in Cluj-Napoca from its features. The plan is to start with **linear regression** and move to more advanced models as the course progresses.

> **For agents:** read this whole file before changing anything. The owner is learning, so keep code simple, well commented and explainable. Prefer numpy/pandas/scikit-learn, and avoid black-box shortcuts unless asked.

## Status

| Step | State |
|------|-------|
| 1. Setup + dataset (Selenium scraper → CSV) | ✅ done |
| 2. Multiple linear regression baseline (`model/linear_regression.py`) | ✅ done: validation RMSE ≈ 111 EUR, R² ≈ 0.68 on 555 flats |
| 3. EDA (distributions, correlations, price per neighborhood) | ⏳ next |
| 4. Feature engineering, regularisation (Ridge/Lasso), log(price) target | ⏳ |
| 5. Non-linear models / small app for predictions | ⏳ |

**Current snapshot (scraped 6–7 Oct 2026):** 3,783 listings scraped → 3,475 in `listings_clean.csv` (after outlier and duplicate removal) → **1,302 in `listings_model.csv`** (22 features, no NaN). Median rent is €600 and the mean is €653 (range €150–3,000). `area_m2` has the strongest correlation with price (0.74), then `rooms` (0.62) and `dist_center_km` (−0.24). `is_agency` is 1 for 98% of rows, so it is nearly constant and almost useless as a feature.

## Layout

```
ImoCJ/
├── scraper/
│   ├── storia_scraper.py   # Selenium: search pages -> listing pages -> raw CSV (resumable)
│   └── build_dataset.py    # raw text CSV -> clean numeric CSV for modelling
├── data/
│   ├── raw/
│   │   ├── storia_index.csv          # 1 row per search result (id, url, price, m2, rooms, neighborhood)
│   │   ├── storia_listings_raw.csv   # 1 row per listing page, raw text values + description + full JSON
│   │   └── scrape_log.txt
│   └── processed/
│       ├── listings_clean.csv        # all cleaned columns, NaN where unknown (for imputation work)
│       └── listings_model.csv        # ← USE THIS FOR MODELLING: no leaks, no NaN, numeric only
├── model/
│   ├── linear_regression.py  # multiple linear regression, mirrors the owner's MATLAB lab (see below)
│   └── figures/              # PNGs saved by the script
└── requirements.txt
```

`data/` and `*.csv` are git-ignored by the root `.gitignore`, so the CSVs exist only locally. Run the scraper again to recreate them. Listings change every day, so a new scrape will not match the old one exactly.

## Data source and how scraping works

- **Source:** [Storia.ro](https://www.storia.ro/ro/rezultate/inchiriere/apartament/cluj/cluj--napoca), apartments for rent in Cluj-Napoca (about 3,800 active listings in Oct 2026).
- **Imobiliare.ro does not work:** it is protected by DataDome and headless Selenium only ever gets the captcha page. OLX.ro loads, but it shares many listings with Storia (same company). It could be added later as a second source, but expect duplicates.
- Storia is a Next.js site. Each page embeds all its data as JSON in `<script id="__NEXT_DATA__">`. Selenium loads the page in a real Chrome, and the script parses that JSON instead of CSS selectors, which is far more robust. For listing pages the useful object is `props.pageProps.ad` (`characteristics`, `additionalInformation`, `location.coordinates`, `description`). For search pages it is `props.pageProps.data.searchAds`.
- Year built, floor details, GPS and amenities only exist on listing pages. That is why there are two phases: index, then details.

```bash
pip install -r requirements.txt
python scraper/storia_scraper.py             # full run, ~1.5 h, resumable (re-run continues)
python scraper/storia_scraper.py --max 1200  # stop at 1200 listings
python scraper/build_dataset.py              # rebuild both processed CSVs
```

Chrome must be installed. Selenium Manager downloads chromedriver automatically.

## `listings_model.csv` (baseline modelling table)

Built from `listings_clean.csv` by `make_model_table()` in `build_dataset.py`:

1. **Drops data leaks and metadata:** `price_per_m2` (computed from the price, so a model would just learn `price_per_m2 * area_m2`), `id`, `url`, `created_at`, `build_year_source`, `n_images`.
2. **Keeps numeric features known for at least 99% of listings** (`MIN_COVERAGE`), **plus the `REQUIRED_FEATURES` `floor`, `build_year` and `has_central_heating`**. These are important for price, so they stay even though many listings lack them. Rows missing any kept feature are dropped (only about 55–60% of listings have all three), so the file has **zero NaN** and can go straight into linear regression: `X = df.drop(columns="price_eur")`, `y = df["price_eur"]`.
3. **Drops text/categorical columns** (`neighborhood`, `heating`, …). Use `listings_clean.csv` and one-hot encoding when you want them.

`building_age` is left out on purpose because it equals `2026 - build_year`, and the two together are perfectly collinear. `building_floors`, `is_top_floor` and `is_ground_floor` are also left out because of too many missing values. Edit `REQUIRED_FEATURES` to change the trade-off between features and rows. The script prints which columns were kept and which were dropped, with their coverage. Note that the `has_*` flags are 0 when the listing does not mention the amenity, which is not the same as confirming it is absent.

## `listings_clean.csv` columns

| Column | Meaning |
|--------|---------|
| `price_eur` | **Target.** Monthly rent in EUR (RON converted at 5.08) |
| `area_m2`, `rooms`, `price_per_m2` | Size. `price_per_m2` is derived from the target: a **data leak, never use it as a feature** |
| `floor`, `building_floors`, `is_ground_floor`, `is_top_floor` | Floor info (ground = 0, basement = -1, 10+ = 11) |
| `build_year`, `build_year_source`, `building_age`, `is_new_building` | Year built comes from the listing field, otherwise from a regex on the description. `build_year_source` is `field`, `description` or `missing` |
| `building_type`, `building_material`, `heating`, `windows_type` | Categorical, `unknown` if missing → one-hot encode |
| `neighborhood`, `latitude`, `longitude`, `dist_center_km` | Location. Distance is measured to Piața Unirii |
| `has_*`, `is_furnished`, `pet_friendly` | 0/1 amenity flags (lift, balcony, terrace, parking, AC, dishwasher, …) |
| `is_agency`, `n_images`, `created_at`, `url`, `id` | Listing metadata (`url` and `id` are for traceability, not features) |

The raw CSV keeps everything as scraped, including the full `description` text and the original JSON (`characteristics_json`, `additional_info_json`). Go back to it if you need a feature that isn't in the clean file.

## Model: `model/linear_regression.py`

It mirrors the owner's MATLAB System Identification lab (`theta = PHI_id \ Y_id`, `MSE_id`/`MSE_val`, `MSE(n)` tuning), so keep that structure and naming when you extend it. The steps:

1. Split 80/20 into identification/validation with `train_test_split(random_state=42)`.
2. Build `PHI = [1, x1..xn]` and solve with `np.linalg.lstsq`, then check that the result equals sklearn `LinearRegression`.
3. Compute MSE, RMSE and R² on both sets and make the validation plots.
4. Show the coefficients, both raw and scaled by 1 std.
5. Plot `MSE(n)` using greedy forward selection of regressors.
6. Predict an example apartment.

Run it with `python model/linear_regression.py` from `ImoCJ/`. Rerun it after rebuilding the data, because the numbers change with the dataset.

## Gotchas and modelling notes

- **Missing values:** `build_year` is missing for many listings and `building_type` and `building_material` are often empty. Impute them (median or a "missing" indicator) instead of dropping rows.
- **Skewed target:** rents are right-skewed. Try `log(price_eur)` as the target for linear regression.
- **Strongest expected predictors:** `area_m2`, `rooms`, `dist_center_km`/`neighborhood`, `is_new_building`/`building_age`, `has_parking`, `is_furnished`.
- **Duplicates:** the same flat is often posted by several agencies. `build_dataset.py` removes rows that have the same price, area, rooms and floor at the same rounded GPS position.
- **Outlier filters** are set in `build_dataset.py` (`PRICE_RANGE`, `AREA_RANGE`, `PPM2_RANGE`).
- **Be polite when scraping:** keep the random delays in the scraper. One full run per day at most is plenty.
