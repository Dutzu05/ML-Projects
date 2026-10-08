"""
CCC Level 5 - A Chirp Disaster: predict the top 50 BOPs by Arrivals for days 731-760.

Only the ranking inside a day matters, and total arrivals swing a lot with season and
wind, so the regression target is each BOP's *share* of the day: arr / mean(arr of day).

Direct multi-horizon forecasting: forecast day 730+k may use arrivals up to day 730,
i.e. lagged by k. One HistGradientBoosting regressor per horizon bucket (1,2,3,4,5,7,
10,14,20,30); day 730+k uses the smallest bucket h >= k, so all its arrival features
are lagged by h days. Near days lean on yesterday's arrivals (top 50 overlap ~88% day
to day), far days on long averages, occupancy (given for the forecast), wind, last
year and the level 4 path graph.

Usage:  python level5.py <data_dir> <out_dir> [--backtest]
  --backtest  also replays the same pipeline on days 641-670, 671-700, 701-730 (slow)
"""
import os, sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BUCKETS = [1, 2, 3, 4, 5, 7, 10, 14, 20, 30]
TOP = 50
LAST_KNOWN = 730
BACKTESTS = [640, 670, 700]


def load(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "in_level-5_1-main.txt"), low_memory=False)
    df.columns = ["day", "bop", "arr", "occ", "wx", "wy", "ins"]
    df["arr"] = pd.to_numeric(df["arr"], errors="coerce")   # "missing" -> NaN
    return df


def graph_stats(data_dir, bops):
    """How often a BOP is flown into, and from how many BOPs (level 4 paths)."""
    paths = pd.read_csv(os.path.join(data_dir, "all_data_from_level_4.in")).iloc[:, 1]
    visits, sources = Counter(), {}
    for p in paths:
        p = [int(x) for x in p.split()]
        visits.update(p[1:])
        for a, b in zip(p, p[1:]):
            if a != b:
                sources.setdefault(b, set()).add(a)
    return (pd.Series(visits).reindex(bops).fillna(0),
            pd.Series({b: len(s) for b, s in sources.items()}).reindex(bops).fillna(0))


class Features:
    """Day x BOP tables; `base` holds features that never use arrivals after day t-365."""

    def __init__(self, df, data_dir):
        wide = lambda c: df.pivot(index="day", columns="bop", values=c)
        self.A, O = wide("arr"), wide("occ").astype(float)
        self.W = df.groupby("day")[["wx", "wy"]].first()
        ins = df.groupby("bop")["ins"].first()
        visits, in_deg = graph_stats(data_dir, O.columns)

        self.S = self.A.div(self.A.mean(1), axis=0)                 # target: share of the day
        self.On = O.div(O.mean(1), axis=0)
        R = self.A / O.clip(lower=1)                                 # arrivals per present bird
        self.Rn = R.div(R.mean(1), axis=0)
        On, S, Rn = self.On, self.S, self.Rn

        base = {
            "occ_n": On, "occ_n_prev": On.shift(1), "occ_n_next": On.shift(-1),
            "occ_n_next2": On.shift(-2), "occ_n_next3": On.shift(-3), "occ_n_prev3": On.shift(3),
            "occ_n_ma7": On.rolling(7, center=True, min_periods=1).mean(), "occ": O,
            "occ_growth": (O - O.shift(1)) / O.shift(1).clip(lower=1),
            "occ_trend7": (O.shift(-3) - O.shift(3)) / O.clip(lower=1),
            "share_365": S.shift(365), "share_365_ma7": S.shift(362).rolling(7, min_periods=1).mean(),
            "ratio_365": Rn.shift(365),
        }
        B = pd.concat({k: self.stack(v) for k, v in base.items()}, axis=1)
        B.index.names = ["day", "bop"]
        self.d = B.index.get_level_values("day").values
        b = B.index.get_level_values("bop").values
        B["ins"] = ins.reindex(b).values
        B["visits"], B["in_deg"] = visits.reindex(b).values, in_deg.reindex(b).values
        B["wx"], B["wy"] = self.W["wx"].reindex(self.d).values, self.W["wy"].reindex(self.d).values
        B["doy"] = (self.d - 1) % 365
        self.base = B
        self.y = self.stack(S).reindex(B.index)

    @staticmethod
    def stack(F):
        return F.stack(future_stack=True)

    def lagged(self, h):
        """Base features + every arrival-derived feature lagged by h days."""
        S, Rn, A = self.S, self.Rn, self.A
        ma = lambda F, n: F.rolling(n, min_periods=1).mean().shift(h)
        lag = {
            "share_h": S.shift(h), "share_h1": S.shift(h + 1), "share_ma7": ma(S, 7),
            "share_ma30": ma(S, 30), "share_ma90": ma(S, 90),
            "ratio_h": Rn.shift(h), "ratio_ma7": ma(Rn, 7), "ratio_ma30": ma(Rn, 30),
            "ratio_ma90": ma(Rn, 90), "zero_ma60": ma(A == 0, 60), "occ_n_h": self.On.shift(h),
        }
        X = self.base.copy()
        for k, v in lag.items():
            X[k] = self.stack(v).reindex(X.index).values
        for k in ["ratio_h", "ratio_ma7", "ratio_ma30", "ratio_ma90"]:
            X[k + "_x_occ"] = X[k] * X["occ_n"]                    # expected share from occupancy
        wl = self.W.shift(h)
        X["wx_h"], X["wy_h"] = wl["wx"].reindex(self.d).values, wl["wy"].reindex(self.d).values
        X["wind_change"] = np.hypot(X["wx"] - X["wx_h"], X["wy"] - X["wy_h"])
        return X


def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=100, random_state=0)


def forecast(F, cutoffs):
    """{cutoff: DataFrame day x bop} of predicted shares for days cutoff+1..cutoff+30."""
    out = {c: [] for c in cutoffs}
    d = F.d
    for i, h in enumerate(BUCKETS):
        lo = BUCKETS[i - 1] + 1 if i else 1
        X = F.lagged(h)
        for cut in cutoffs:
            tr = (d > 365) & (d <= cut) & F.y.notna().values   # need a year of history
            reg = model().fit(X[tr], F.y[tr])
            te = (d >= cut + lo) & (d <= cut + h)
            out[cut].append(pd.Series(reg.predict(X[te]), X.index[te]).unstack("bop"))
        print(f"  horizon bucket {lo}-{h} done", flush=True)
    return {c: pd.concat(v).sort_index() for c, v in out.items()}


def top(row):
    return list(row.nlargest(TOP).index)


def accuracy(pred, A):
    return np.array([len(set(top(pred.loc[d])) & set(top(A.loc[d]))) / TOP for d in pred.index])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data_dir, out_dir = (args + [".", "."])[:2]
    os.makedirs(out_dir, exist_ok=True)
    F = Features(load(data_dir), data_dir)

    cutoffs = [LAST_KNOWN] + (BACKTESTS if "--backtest" in sys.argv else [])
    preds = forecast(F, cutoffs)

    scores = []
    for cut in cutoffs[1:]:
        acc = accuracy(preds[cut], F.A)
        scores.append(acc.mean())
        print(f"backtest days {cut + 1}-{cut + 30}: {acc.mean():.3f} "
              f"(days 1-10: {acc[:10].mean():.2f}, days 11-30: {acc[10:].mean():.2f})")
    if scores:
        print(f"mean backtest accuracy: {np.mean(scores):.3f}  (required 0.50)")

    pred = preds[LAST_KNOWN]
    out = "Day,Top 50 Arrivals BOPs\n" + "".join(
        f"{d},{' '.join(map(str, top(pred.loc[d])))}\n" for d in pred.index)
    path = os.path.join(out_dir, "out_level-5_1-main.txt")
    open(path, "w").write(out)
    print(f"wrote {path} ({len(pred)} days)")
