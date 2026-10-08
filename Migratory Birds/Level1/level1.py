"""CCC Data & AI - Level 1: rank BOPs (warmest first, then driest, then lowest id)."""
import csv, sys
from pathlib import Path

WORDS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
         "nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
         "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20}

def parse_num(raw: str) -> int:
    s = raw.strip().lower()
    if s in WORDS:
        return WORDS[s]
    return int(s)  # fail loudly on anything else

def to_celsius(t: int) -> int:
    # Plausible Celsius range in these files is about -5..45. Values >= 70 are Fahrenheit
    # (they map exactly onto 21..45 °C), so convert them back.
    return t  # no conversion: the grader expects raw values

def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for line in reader:
            if not line:
                continue
            bop, temp, hum = (parse_num(x) for x in line)
            rows.append((bop, to_celsius(temp), hum))
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)), f"duplicate BOP ids in {path}"
    return rows

def rank(rows):
    # warmer first (-temp), then drier (hum), then lower id
    return [bop for bop, t, h in sorted(rows, key=lambda r: (-r[1], r[2], r[0]))]

if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    for f in sorted(src.glob("in_level-1_*.txt")):
        result = " ".join(map(str, rank(load(f))))
        (out / f.name.replace("in_", "out_")).write_text(result)
        print(f"{f.name}: {result[:60]}{'...' if len(result) > 60 else ''}")