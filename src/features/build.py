"""Features (sentiment + price ONLY) and targets.

THE LEAKAGE RULE: row n is the prediction made at the START of bar n. It may only see
information up to the CLOSE of bar n-1. One global `.shift(1)` enforces that for every
feature. Calendar features are known in advance, so they are added AFTER the shift.

Targets: y_h = log(close[n+h] / close[n-1]), h = 0..5   (h=0 -> return of the current bar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZONS, PROC, TARGETS
from src.utils.log import get_logger

log = get_logger(__name__)

SENT_BASE = ["sent_mean", "w_follow", "w_engage", "pct_pos"]
LAGS = [1, 2, 3, 6, 12, 24]
NON_FEATURES = TARGETS + ["hour", "close_prev"]


def _z(s: pd.Series, window: int = 168) -> pd.Series:
    """Rolling z-score vs the trailing week (min 24 obs so early rows are not all lost)."""
    r = s.rolling(window, min_periods=24)
    return (s - r.mean()) / r.std()


def build(master: pd.DataFrame, add_future_row: bool = False) -> pd.DataFrame:
    """add_future_row=True appends one empty row for the NEXT bar: its features are the
    latest known information and its targets are NaN. That is the row we forecast 'now'."""
    m = master.sort_values("hour").reset_index(drop=True).copy()
    if add_future_row:
        m = pd.concat(
            [m, pd.DataFrame({"hour": [m.hour.iloc[-1] + pd.Timedelta(hours=1)]})],
            ignore_index=True,
        )

    ret = np.log(m.close).diff()
    F: dict[str, pd.Series] = {}

    # price features: lag1 = the most recent completed bar
    for lag in LAGS:
        F[f"ret_lag{lag}"] = ret.shift(lag - 1)
    F["vol_6"] = ret.rolling(6, min_periods=4).std()
    F["vol_24"] = ret.rolling(24, min_periods=12).std()
    F["mom_24"] = np.log(m.close / m.close.shift(24))
    F["volume_z"] = _z(m.volume)

    # sentiment features
    for col in SENT_BASE:
        for lag in LAGS:
            F[f"{col}_lag{lag}"] = m[col].shift(lag - 1)
        F[f"{col}_ma6"] = m[col].rolling(6, min_periods=3).mean()
        F[f"{col}_ma24"] = m[col].rolling(24, min_periods=12).mean()
        F[f"{col}_z168"] = _z(m[col])
        F[f"{col}_d1"] = m[col].diff()

    F["tweets_log"] = np.log1p(m.n_tweets)
    F["tweets_z168"] = _z(m.n_tweets)
    F["no_tweets"] = m.no_tweets

    X = pd.DataFrame(F).shift(1)                        # <<< THE LEAKAGE GUARD <<<
    hr = m.hour.dt.hour
    X["hr_sin"] = np.sin(2 * np.pi * hr / 24)
    X["hr_cos"] = np.cos(2 * np.pi * hr / 24)
    X["dow"] = m.hour.dt.day_of_week
    feat_cols = list(X.columns)

    base = np.log(m.close.shift(1))                     # close of bar n-1
    for h in HORIZONS:
        X[f"y{h}"] = np.log(m.close.shift(-h)) - base
    X["hour"] = m.hour
    X["close_prev"] = m.close.shift(1)

    X = X.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols).reset_index(drop=True)
    return X


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def split_features(cols: list[str]) -> tuple[list[str], list[str]]:
    """(price/calendar columns, sentiment columns) for the price-only ablation."""
    prefixes = tuple(SENT_BASE) + ("tweets_", "no_tweets")
    sent = [c for c in cols if c.startswith(prefixes)]
    return [c for c in cols if c not in sent], sent


def main() -> None:
    master = pd.read_parquet(PROC / "master.parquet")
    df = build(master, add_future_row=True)
    df.to_parquet(PROC / "features.parquet", index=False)
    log.info(
        "features: %d rows (incl. 1 forecast row), %d feature columns",
        len(df), len(feature_columns(df)),
    )


if __name__ == "__main__":
    main()
