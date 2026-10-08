"""
CCC Level 5 - A Chirp Disaster: predict the top 50 BOPs by Arrivals for days 731-760.

The data is a flow simulation, and occupancy is given for the forecast days, so most of
the answer follows from bookkeeping instead of a black-box regressor:

  departures  D[t] = O[t-1] + A[t] - O[t]        (always >= 0 in the data)
  arrivals    A[t] = (O[t] - O[t-1]) + D[t]

- O[t] - O[t-1] is known exactly for the forecast.
- Today's departures are tomorrow's arrivals (sum D[t] = sum A[t+1]), so
  sum A[t] + sum O[t-1] is constant (1,204,141 +- 7 birds). The daily total of arrivals,
  and hence the daily departure rate rho[t] = sum D[t] / sum O[t-1], is exact for the
  forecast too.
- What is left is each BOP's departure rate relative to the day: rel = (D/O_prev)/rho.
  With the true rel the top-50 accuracy is ~0.98, so all the work is forecasting rel.

rel behaves like a mean-reverting process (lag-1 corr 0.92, lag-30 0.11). Per horizon
k = 1..30 a linear model maps the recent rel averages (1, 3, 7, 14, 30, 90 days) and
the crowding (occupancy vs. its 60-day mean, at the cutoff and on the target day) to
rel[c+k]. It is trained on every past cutoff and weighted by O_prev, since only
the big BOPs can reach the top 50.

The final ranking blends this 50/50 with the gradient-boosting model in level5_gbm.py
(both as share of the day's mean). Backtest: flow 0.707, gbm 0.726, blend 0.739.

Usage:  python level5.py <data_dir> <out_dir> [--backtest]
  --backtest  also replays the pipeline on days 551-580, ..., 701-730 (slow)
"""
import os, sys, warnings

import numpy as np
import pandas as pd

import level5_gbm

TOP = 50
HORIZON = 30
LAST_KNOWN = 730
BACKTESTS = [550, 580, 610, 640, 670, 700]
TRAIN_FROM, TRAIN_STEP = 100, 3        # past cutoffs used to fit the horizon models
BLEND = 0.5                            # weight of the flow model vs. level5_gbm


def load(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "in_level-5_1-main.txt"), low_memory=False)
    df.columns = ["day", "bop", "arr", "occ", "wx", "wy", "ins"]
    df["arr"] = pd.to_numeric(df["arr"], errors="coerce")   # "missing" -> NaN
    wide = lambda c: df.pivot(index="day", columns="bop", values=c).astype(float)
    return wide("arr"), wide("occ")


class Flow:
    """Day x BOP arrays; row i is day i+1, so day d is row d-1."""

    def __init__(self, A, O):
        self.bops = A.columns
        self.A, self.O = A.values, O.values
        self.Op = O.shift(1).values
        D = self.Op + self.A - self.O
        rho = np.nansum(D, 1) / np.nansum(self.Op, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rel = D / np.clip(self.Op, 1, None) / rho[:, None]
        self.total = np.nansum(self.A + self.Op, 1)     # birds in the system, constant

    def row(self, day):
        return day - 1

    def mean(self, X, first, last):
        """nan-mean of X over days first..last."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns
            return np.nanmean(X[self.row(first):self.row(last) + 1], 0)

    def features(self, c, k):
        """Features for predicting rel on day c+k from data up to day c."""
        rel = [self.mean(self.rel, c - n + 1, c) for n in (1, 3, 7, 14, 30, 90)]
        o_mu = self.mean(self.Op, c - 59, c)
        crowd_t = self.Op[self.row(c + k)] / o_mu - 1
        crowd_c = self.Op[self.row(c)] / o_mu - 1
        F = np.c_[np.ones(len(self.bops)), *rel, crowd_t, crowd_c, crowd_t * rel[2]]
        return np.nan_to_num(F, nan=1.0)

    def forecast(self, c):
        """Predicted arrivals, shape (HORIZON, n_bops), for days c+1..c+HORIZON."""
        days = np.arange(c + 1, c + HORIZON + 1)
        O, Op = self.O[days - 1], self.Op[days - 1]

        # daily departure rate from conservation of birds: sum A[t] = total - sum O[t-1]
        total = np.median(self.total[self.row(c - 29):self.row(c) + 1])
        rho = (total - O.sum(1)) / Op.sum(1)        # sum D[t] / sum O[t-1], sum D[t] = sum A[t+1]

        rel = np.zeros_like(O)
        for k in range(1, HORIZON + 1):
            X, y, w = [], [], []
            for cc in range(TRAIN_FROM, c - k + 1, TRAIN_STEP):
                X.append(self.features(cc, k))
                y.append(self.rel[self.row(cc + k)])
                w.append(self.Op[self.row(cc + k)])
            X, y, w = np.vstack(X), np.concatenate(y), np.concatenate(w)
            ok = np.isfinite(y) & np.isfinite(w)
            coef = np.linalg.lstsq(X[ok] * w[ok, None], y[ok] * w[ok], rcond=None)[0]
            rel[k - 1] = self.features(c, k) @ coef

        return (O - Op) + rho[:, None] * rel * Op


def top(row, bops):
    return list(bops[np.argsort(-row, kind="stable")[:TOP]])


def accuracy(pred, flow, c):
    return np.array([len(set(top(p, flow.bops)) & set(top(flow.A[c + i], flow.bops))) / TOP
                     for i, p in enumerate(pred)])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data_dir, out_dir = (args + [".", "."])[:2]
    os.makedirs(out_dir, exist_ok=True)
    flow = Flow(*load(data_dir))

    cutoffs = [LAST_KNOWN] + (BACKTESTS if "--backtest" in sys.argv else [])
    F = level5_gbm.Features(level5_gbm.load(data_dir), data_dir)
    gbm = level5_gbm.forecast(F, cutoffs)
    share = lambda x: x / np.nanmean(x, 1, keepdims=True)
    preds = {c: BLEND * share(flow.forecast(c))
                + (1 - BLEND) * share(gbm[c].reindex(columns=flow.bops).values) for c in cutoffs}

    scores = []
    for c in cutoffs[1:]:
        acc = accuracy(preds[c], flow, c)
        scores.append(acc.mean())
        print(f"backtest days {c + 1}-{c + HORIZON}: {acc.mean():.3f} "
              f"(days 1-10: {acc[:10].mean():.2f}, days 11-30: {acc[10:].mean():.2f})", flush=True)
    if scores:
        print(f"mean backtest accuracy: {np.mean(scores):.3f}  (required 0.50)")

    pred = preds[LAST_KNOWN]
    out = "Day,Top 50 Arrivals BOPs\n" + "".join(
        f"{LAST_KNOWN + 1 + i},{' '.join(map(str, top(p, flow.bops)))}\n" for i, p in enumerate(pred))
    path = os.path.join(out_dir, "out_level-5_1-main.txt")
    open(path, "w").write(out)
    print(f"wrote {path} ({len(pred)} days)")
