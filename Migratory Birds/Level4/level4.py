"""
CCC Level 4 - Going Global: predict the species of the "missing" flocks.

Pipeline: load birds -> engineer one feature row per flock -> Random Forest trained
on the labeled flocks (cross-validated, F1 macro) -> predict the missing flocks.

Usage:  python level4.py <data_dir> <out_dir>
"""
import glob, os, sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report

F_THRESHOLD = 50   # same °F-bug as before: anything above is °F
HOT = 30           # °C, "hot regions" for the Hurracurra worm


def load(path):
    df = pd.read_csv(path)
    df.columns = ["flock", "path", "species"]
    df["path"] = df["path"].map(lambda s: tuple(int(x) for x in s.split()))
    return df


def load_temps(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "all_data_from_level_1.in"))
    bop, temp = df.columns[0], df.columns[1]           # header has a mojibake °C
    t = pd.to_numeric(df[temp], errors="coerce").astype(float)
    t[t > F_THRESHOLD] = (t[t > F_THRESHOLD] - 32) * 5 / 9
    return dict(zip(pd.to_numeric(df[bop], errors="coerce"), t))


def common_prefix(paths):
    n = 0
    for col in zip(*paths):
        if len(set(col)) > 1:
            break
        n += 1
    return n


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def flock_features(paths, temps, palin_nodes):
    lens = np.array([len(p) for p in paths], float)
    visits = [b for p in paths for b in p]
    known = [temps[b] for b in visits if b in temps and not np.isnan(temps[b])]
    nodes = set(visits)
    pairs = list(combinations(paths, 2)) or [(paths[0], paths[0])]
    # how often each node is shared by the birds of the flock
    share = pd.Series([b for p in paths for b in set(p)]).value_counts() / len(paths)
    trailing = [len(p) - len(tuple(pd.Series(p[::-1]).loc[lambda s: s != p[-1]])) for p in paths]
    return {
        "n_birds": len(paths),
        "n_unique_paths": len(set(paths)) / len(paths),
        "identical": float(len(set(paths)) == 1),                  # Blackfinch / Bluetit
        "palin": np.mean([p == p[::-1] for p in paths]),            # Bluetit flies back the same way
        "closed": np.mean([p[0] == p[-1] for p in paths]),
        "same_start": float(len({p[0] for p in paths}) == 1),
        "len_mean": lens.mean(), "len_std": lens.std(), "len_min": lens.min(),
        "prefix_rel": common_prefix(paths) / lens.min(),            # Goldhammer: together, then split
        "suffix_rel": common_prefix([p[::-1] for p in paths]) / lens.min(),
        "prefix_abs": common_prefix(paths),
        "jaccard": np.mean([jaccard(a, b) for a, b in pairs]),
        "pos_equal": np.mean([np.mean([x == y for x, y in zip(a, b)]) for a, b in pairs]),
        "self_revisit": np.mean([1 - len(set(p)) / len(p) for p in paths]),
        "trailing_rep": np.mean(trailing),                          # e.g. "379 379 379 ..."
        "core_frac": (share >= 1).mean(),                           # nodes every bird visits (nest)
        "shared_frac": (share > 1 / len(paths)).mean(),
        "nodes_per_bird": len(nodes) / len(paths),
        "in_palin": np.mean([b in palin_nodes for b in nodes]),     # Wolfthroat lurks on Bluetit routes
        "temp_mean": np.mean(known) if known else np.nan,           # Hurracurra likes heat
        "temp_min": np.min(known) if known else np.nan,
        "temp_max": np.max(known) if known else np.nan,
        "hot_frac": np.mean([t >= HOT for t in known]) if known else np.nan,
    }


def build(birds, temps):
    # nodes of all routes that fly back the same way (label-free proxy for Bluetit routes)
    palin_nodes = {b for p in birds["path"] if p == p[::-1] and len(p) > 5 for b in p}
    rows, labels = {}, {}
    for fid, g in birds.groupby("flock"):
        rows[fid] = flock_features(list(g["path"]), temps, palin_nodes)
        labels[fid] = g["species"].iloc[0]
    X = pd.DataFrame.from_dict(rows, orient="index").fillna(-1)
    return X, pd.Series(labels)


def model():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def train(path, temps):
    """Fit on every labeled flock of the main file (CV-scored first)."""
    X, y = build(load(path), temps)
    Xtr, ytr = X[y != "missing"], y[y != "missing"]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    pred = cross_val_predict(model(), Xtr, ytr, cv=cv)
    print(f"CV F1 macro: {f1_score(ytr, pred, average='macro'):.4f}")
    print(classification_report(ytr, pred, digits=3))
    clf = model().fit(Xtr, ytr)
    imp = pd.Series(clf.feature_importances_, X.columns).sort_values(ascending=False)
    print("top features:", ", ".join(f"{k}={v:.3f}" for k, v in imp.head(8).items()))
    return clf


def predict(clf, path, temps):
    X, y = build(load(path), temps)
    X = X[y == "missing"]
    return pd.Series(clf.predict(X), X.index).sort_index()


data_dir, out_dir = (sys.argv[1:3] + [".", "."])[:2]
os.makedirs(out_dir, exist_ok=True)
temps = load_temps(data_dir)
clf = train(os.path.join(data_dir, "in_level-4_1-main.txt"), temps)
for f in sorted(glob.glob(os.path.join(data_dir, "in_level-4_*.txt"))):
    name = f.split("in_level-4_")[1]
    res = predict(clf, f, temps)
    txt = "Flock ID,Species\n" + "".join(f"{k},{v}\n" for k, v in res.items())
    if "example" in name:
        exp = open(os.path.join(data_dir, "out_level-4_0-example.txt")).read()
        print("example matches:", txt.strip() == exp.strip()); continue
    open(os.path.join(out_dir, f"out_level-4_{name}"), "w").write(txt)
    print("==", name, f"({len(res)} missing flocks)")
    print(res.value_counts().to_string())
