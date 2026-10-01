"""
CCC Level 2 - Bird Love Score
Pipeline: load -> clean (°F fix) -> merge -> validate model -> predict -> write CSVs

Usage:  python bird_love.py <data_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

FEATURES = ["veg", "insects", "light", "temp", "hum"]
TARGET = "score"
F_THRESHOLD = 50  # data has a gap between 45 and 70 -> anything above is °F


# ---------- 1. Loading ----------
def load_level1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df.columns = ["BOP", "temp", "hum"]
    return df


def load_level2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", na_values="missing")
    df.columns = ["BOP", "veg", "insects", "light", TARGET]
    return df


# ---------- 2. Cleaning ----------
def fix_fahrenheit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp"] = df["temp"].astype(float)  # int column can't hold converted floats
    is_f = df["temp"] > F_THRESHOLD
    df.loc[is_f, "temp"] = (df.loc[is_f, "temp"] - 32) * 5 / 9
    print(f"[clean] converted {is_f.sum()} temperatures from °F to °C")
    return df


# ---------- 3. Model ----------
def evaluate(model, X, y) -> float:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return scores.mean()


def main(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    weather = fix_fahrenheit(load_level1(next(data_dir.glob("*level_1*.in"))))

    inputs = {p: load_level2(p).merge(weather, on="BOP", how="left")
              for p in sorted(data_dir.glob("*in_level-2_*.txt"))}

    # Train on every row that HAS a label, across all files (same underlying world)
    train = pd.concat(inputs.values()).dropna(subset=[TARGET]).drop_duplicates("BOP")
    X, y = train[FEATURES], train[TARGET]

    baseline_rmse = np.sqrt(((y - y.mean()) ** 2).mean())
    model = LinearRegression()
    print(f"[eval] predict-the-mean RMSE: {baseline_rmse:.3f}")
    print(f"[eval] linear regression CV RMSE: {evaluate(model, X, y):.3f}")

    model.fit(X, y)
    print("[model] coefficients:", dict(zip(FEATURES, model.coef_.round(3))),
          "intercept:", round(model.intercept_, 3))

    # ---------- 4. Predict & write ----------
    for path, df in inputs.items():
        missing = df[df[TARGET].isna()]
        preds = model.predict(missing[FEATURES])
        out = pd.DataFrame({"BOP": missing["BOP"],
                            "Bird Love Score [<3]": np.round(preds, 2)})
        out_path = out_dir / path.name.replace("in_", "out_")
        out.to_csv(out_path, index=False, float_format="%.2f")
        print(f"[write] {out_path.name}: {len(out)} predictions")


if __name__ == "__main__":
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
    main(data, out)