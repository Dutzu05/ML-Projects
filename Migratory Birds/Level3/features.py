"""
CCC Level 3 - per-flock route features used by level3.classify()
"""
import os

import pandas as pd

F_THRESHOLD = 50  # same °F-bug as level 2: anything above is °F


def load(path):
    """One row per bird: flock id + list of visited BOPs."""
    df = pd.read_csv(path)
    df.columns = ["flock", "path"]
    df["path"] = df["path"].map(lambda s: tuple(int(x) for x in s.split()))
    return df


def load_temps(data_dir):
    """BOP -> temperature in °C (fixes the °F rows)."""
    df = pd.read_csv(os.path.join(data_dir, "all_data_from_level_1.in"))
    bop, temp = df.columns[0], df.columns[1]  # header has a mojibake °C
    t = df[temp].astype(float)
    t[t > F_THRESHOLD] = (t[t > F_THRESHOLD] - 32) * 5 / 9
    return dict(zip(df[bop], t))


def common_prefix(paths):
    n = 0
    for col in zip(*paths):
        if len(set(col)) > 1:
            break
        n += 1
    return n


def features(birds, temps):
    """One row per flock with the features classify() looks at."""
    rows = {}
    for fid, g in birds.groupby("flock"):
        paths = list(g["path"])
        visits = [b for p in paths for b in p]
        known = [temps[b] for b in visits if b in temps]
        rows[fid] = {
            "identical": len(set(paths)) == 1,                 # whole flock flies one route
            "palin": all(p == p[::-1] for p in paths),         # flies back the same way
            "nodes": frozenset(visits),                        # BOPs the flock touches
            "prefix": common_prefix(paths),                    # shared start before splitting
            "mean_temp": sum(known) / len(known) if known else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
