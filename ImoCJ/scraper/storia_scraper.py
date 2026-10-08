"""
Selenium scraper for apartment rentals in Cluj-Napoca from Storia.ro.

Storia is a Next.js site: every page embeds its full data as JSON inside
<script id="__NEXT_DATA__">. We let Selenium (a real Chrome) load the page,
then read that JSON instead of fragile CSS selectors.

Two phases (both resumable - re-running skips work that is already saved):
  1. index   - loop through search result pages -> data/raw/storia_index.csv
               (id, url, price, m2, rooms, neighborhood ... one row per listing)
  2. details - open every listing page -> data/raw/storia_listings_raw.csv
               (all characteristics incl. year built, floor, GPS, amenities, description)

Usage (from the ImoCJ folder):
  python scraper/storia_scraper.py                 # both phases, all listings
  python scraper/storia_scraper.py --max 1200      # stop after 1200 detail pages
  python scraper/storia_scraper.py --phase details # only (continue) phase 2
  python scraper/storia_scraper.py --show          # visible browser (debugging)
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import (InvalidSessionIdException, NoSuchWindowException,
                                        WebDriverException)
from selenium.webdriver.chrome.options import Options

BASE = "https://www.storia.ro"
SEARCH_URL = BASE + "/ro/rezultate/inchiriere/apartament/cluj/cluj--napoca"
AD_URL = BASE + "/ro/oferta/{slug}"
PAGE_SIZE = 72

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INDEX_CSV = RAW_DIR / "storia_index.csv"
DETAILS_CSV = RAW_DIR / "storia_listings_raw.csv"

NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)

INDEX_FIELDS = [
    "id", "slug", "url", "title", "price", "currency", "area_m2", "rooms",
    "floor", "neighborhood", "tags", "is_private_owner", "advertiser_type",
    "date_created", "search_page",
]

DETAIL_FIELDS = [
    "id", "url", "title", "scraped_at", "created_at", "modified_at",
    # core target + size
    "price", "currency", "area_m2", "rooms",
    # building
    "floor", "building_floors", "build_year", "building_type",
    "building_material", "construction_status", "windows_type", "heating", "lift",
    # location
    "neighborhood", "location_detail", "street", "latitude", "longitude",
    # amenities (pipe-separated lists of raw codes)
    "extras", "equipment", "security", "media",
    # misc
    "free_from", "advertiser_type", "agency_name", "n_images", "description",
    # everything else, untouched, so no information is lost
    "characteristics_json", "additional_info_json",
]


def make_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.page_load_strategy = "eager"  # don't wait for images/ads
    opts.add_argument("--blink-settings=imagesEnabled=false")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--lang=ro-RO")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(options=opts)  # Selenium Manager fetches chromedriver
    driver.set_page_load_timeout(40)
    return driver


def load_next_data(driver, url, retries=3):
    """Open url and return the parsed __NEXT_DATA__ JSON (or None)."""
    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            for _ in range(20):  # wait up to ~10s for the script tag
                m = NEXT_DATA_RE.search(driver.page_source)
                if m:
                    return json.loads(m.group(1))
                time.sleep(0.5)
        except (InvalidSessionIdException, NoSuchWindowException):
            raise  # browser is dead (e.g. laptop slept) - caller restarts it
        except (WebDriverException, json.JSONDecodeError) as e:
            print(f"  ! attempt {attempt} failed for {url}: {type(e).__name__}")
        time.sleep(3 * attempt)
    return None


def read_ids(path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def open_writer(path, fields):
    new = not path.exists()
    f = open(path, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    return f, w


def district_of(locations):
    """Deepest location below city level (e.g. 'Manastur', 'Zorilor')."""
    below_city = [l for l in locations or []
                  if l.get("locationLevel") not in ("county", "county_capital", "city")]
    return below_city[0]["name"] if below_city else ""


# ---------------------------------------------------------------- phase 1
def scrape_index(driver, max_pages=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen = read_ids(INDEX_CSV)
    f, w = open_writer(INDEX_CSV, INDEX_FIELDS)
    page, total_pages = 1, None
    try:
        while total_pages is None or page <= total_pages:
            url = f"{SEARCH_URL}?limit={PAGE_SIZE}&page={page}&by=LATEST&direction=DESC"
            data = load_next_data(driver, url)
            if data is None:
                print(f"page {page}: no data, stopping index phase")
                break
            search = data["props"]["pageProps"]["data"]["searchAds"]
            total_pages = search["pagination"]["totalPages"]
            if max_pages:
                total_pages = min(total_pages, max_pages)
            new = 0
            for it in search["items"]:
                sid = str(it["id"])
                if sid in seen:
                    continue
                seen.add(sid)
                new += 1
                price = it.get("totalPrice") or it.get("rentPrice") or {}
                w.writerow({
                    "id": sid,
                    "slug": it.get("slug", ""),
                    "url": AD_URL.format(slug=it.get("slug", "")),
                    "title": it.get("title", ""),
                    "price": price.get("value", ""),
                    "currency": price.get("currency", ""),
                    "area_m2": it.get("areaInSquareMeters", ""),
                    "rooms": it.get("roomsNumber", ""),
                    "floor": it.get("floorNumber", "") or "",
                    "neighborhood": district_of(
                        it.get("location", {}).get("reverseGeocoding", {}).get("locations")),
                    "tags": "|".join(t["value"] for t in it.get("tags") or []),
                    "is_private_owner": it.get("isPrivateOwner", ""),
                    "advertiser_type": it.get("extendedAdvertiserType", ""),
                    "date_created": it.get("dateCreated", ""),
                    "search_page": page,
                })
            f.flush()
            print(f"index page {page}/{total_pages}: +{new} new (total {len(seen)})")
            page += 1
            time.sleep(random.uniform(1.0, 2.5))
    finally:
        f.close()
    return len(seen)


# ---------------------------------------------------------------- phase 2
def info_values(info_list, label):
    """additionalInformation values look like 'extras_types::balcony' -> 'balcony'."""
    for item in info_list:
        if item.get("label") == label:
            return [v.split("::", 1)[-1] for v in item.get("values", [])]
    return []


def parse_ad(ad, url):
    ch = {c["key"]: c.get("value", "") for c in ad.get("characteristics") or []}
    ch_currency = {c["key"]: c.get("currency", "") for c in ad.get("characteristics") or []}
    info = (ad.get("topInformation") or []) + (ad.get("additionalInformation") or [])
    loc = ad.get("location") or {}
    coords = loc.get("coordinates") or {}
    address = loc.get("address") or {}
    locations = (loc.get("reverseGeocoding") or {}).get("locations") or []
    below_city = [l["name"] for l in locations
                  if l.get("locationLevel") not in ("county", "county_capital", "city")]
    street = address.get("street") or {}
    agency = ad.get("agency") or {}
    owner = ad.get("owner") or {}
    desc = re.sub(r"<[^>]+>", " ", ad.get("description") or "")
    desc = re.sub(r"\s+", " ", desc).strip()

    return {
        "id": str(ad.get("id", "")),
        "url": url,
        "title": ad.get("title", ""),
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_at": ad.get("createdAt", ""),
        "modified_at": ad.get("modifiedAt", ""),
        "price": ch.get("price", ""),
        "currency": ch_currency.get("price", ""),
        "area_m2": ch.get("m", ""),
        "rooms": ch.get("rooms_num", ""),
        "floor": ch.get("floor_no", ""),
        "building_floors": ch.get("building_floors_num", ""),
        "build_year": ch.get("build_year", ""),
        "building_type": ch.get("building_type", ""),
        "building_material": ch.get("building_material", ""),
        "construction_status": ch.get("construction_status", ""),
        "windows_type": ch.get("windows_type", ""),
        "heating": ch.get("heating", ""),
        "lift": "|".join(info_values(info, "lift")),
        "neighborhood": below_city[0] if below_city else "",
        "location_detail": " > ".join(below_city),
        "street": street.get("name", "") if isinstance(street, dict) else str(street),
        "latitude": coords.get("latitude", ""),
        "longitude": coords.get("longitude", ""),
        "extras": "|".join(info_values(info, "extras_types")),
        "equipment": "|".join(info_values(info, "equipment_types")),
        "security": "|".join(info_values(info, "security_types")),
        "media": "|".join(info_values(info, "media_types")),
        "free_from": ch.get("free_from", ""),
        "advertiser_type": "|".join(info_values(info, "advertiser_type")),
        "agency_name": agency.get("name") or owner.get("name") or "",
        "n_images": len(ad.get("images") or []),
        "description": desc,
        "characteristics_json": json.dumps(ad.get("characteristics") or [], ensure_ascii=False),
        "additional_info_json": json.dumps(info, ensure_ascii=False),
    }


def restart_driver(driver):
    try:
        driver.quit()
    except WebDriverException:
        pass
    return make_driver(driver_headless)


def scrape_details(driver, max_items=None, restart_every=400):
    with open(INDEX_CSV, encoding="utf-8", newline="") as fi:
        index = list(csv.DictReader(fi))
    done = read_ids(DETAILS_CSV)
    todo = [r for r in index if r["id"] not in done]
    if max_items is not None:
        todo = todo[:max(0, max_items - len(done))]
    print(f"details: {len(done)} already saved, {len(todo)} to go")

    f, w = open_writer(DETAILS_CSV, DETAIL_FIELDS)
    ok = fail = 0
    t0 = time.time()
    try:
        for i, row in enumerate(todo, 1):
            try:
                data = load_next_data(driver, row["url"])
            except (InvalidSessionIdException, NoSuchWindowException):
                print("  ! browser session lost - restarting Chrome")
                driver = restart_driver(driver)
                data = load_next_data(driver, row["url"])
            ad = (data or {}).get("props", {}).get("pageProps", {}).get("ad")
            if not ad:  # listing removed / expired since indexing
                fail += 1
            else:
                w.writerow(parse_ad(ad, row["url"]))
                ok += 1
            if i % 25 == 0 or i == len(todo):
                f.flush()
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate / 60
                print(f"  {i}/{len(todo)}  ok={ok} fail={fail}  {rate:.2f}/s  eta {eta:.0f} min")
            if i % restart_every == 0:  # fresh browser keeps memory low
                driver = restart_driver(driver)
            time.sleep(random.uniform(0.4, 1.2))
    finally:
        f.close()
    return driver


driver_headless = True


def main():
    global driver_headless
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["all", "index", "details"], default="all")
    ap.add_argument("--max", type=int, default=None, help="max detail listings to have in total")
    ap.add_argument("--max-pages", type=int, default=None, help="max search pages to index")
    ap.add_argument("--show", action="store_true", help="show the browser window")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    driver_headless = not args.show

    driver = make_driver(driver_headless)
    try:
        if args.phase in ("all", "index"):
            n = scrape_index(driver, args.max_pages)
            print(f"index done: {n} listings in {INDEX_CSV.relative_to(ROOT)}")
        if args.phase in ("all", "details"):
            driver = scrape_details(driver, args.max)
            print(f"details done: {len(read_ids(DETAILS_CSV))} listings in {DETAILS_CSV.relative_to(ROOT)}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
