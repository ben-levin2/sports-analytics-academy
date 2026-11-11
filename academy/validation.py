"""Concepts 25 (calibration and Brier score) and 26 (time-based cross-validation).

Pure functions, used by lessons L3.1-L3.2 and by every automated Production Candidate check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 26: time-based validation
def rolling_origin(df: pd.DataFrame, time_col: str = "season", min_train_periods: int = 1):
    """Yield (T, train_df, test_df): train = every period before T, test = period T (expanding window)."""
    periods = sorted(df[time_col].unique())
    for T in periods[min_train_periods:]:
        yield T, df[df[time_col] < T], df[df[time_col] == T]


def no_leakage(train: pd.DataFrame, test: pd.DataFrame, time_col: str = "season") -> bool:
    """True when every training row is strictly earlier than every test row."""
    return train.empty or test.empty or train[time_col].max() < test[time_col].min()


# ---------------------------------------------------------------- 25: calibration
def brier(p, y) -> float:
    """Mean squared gap between predicted probability and what happened (0/1). Lower is better."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def brier_skill(p, y, p_ref) -> float:
    """1 - Brier(model) / Brier(reference). Above 0 = better than the reference forecast."""
    ref = brier(np.broadcast_to(p_ref, np.shape(y)), y)
    return float(1 - brier(p, y) / ref) if ref > 0 else 0.0


def reliability_table(p, y, bins: int = 5) -> pd.DataFrame:
    """Group forecasts into equal-size bins; compare mean forecast with the observed frequency."""
    d = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    if d["p"].nunique() <= 1:
        d["bin"] = 0
    else:   # tied forecasts always share a bin, so ties can't fake miscalibration
        d["bin"] = pd.qcut(d["p"], bins, labels=False, duplicates="drop")
    return (d.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
             .reset_index(drop=True))


def ece(p, y, bins: int = 5) -> float:
    """Expected calibration error: average |forecast - observed| across bins, weighted by bin size."""
    t = reliability_table(p, y, bins)
    return float(np.average((t["predicted"] - t["observed"]).abs(), weights=t["n"]))


def auc(score, y) -> float:
    """Discrimination: chance a random 'yes' case is scored above a random 'no' case."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, score)) if 0 < y.mean() < 1 else float("nan")
