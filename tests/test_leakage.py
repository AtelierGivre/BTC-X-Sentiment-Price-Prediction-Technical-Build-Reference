"""THE most important test: row n's features must not change if data at bar n or later changes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import build, feature_columns


def fake_master(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    m = pd.DataFrame({
        "hour": hours,
        "close": close,
        "volume": rng.lognormal(5, 0.3, n),
        "n_tweets": rng.poisson(20, n).astype(float),
        "no_tweets": 0.0,
    })
    for col in ["sent_mean", "w_follow", "w_engage", "pct_pos"]:
        m[col] = rng.normal(0, 1, n)
    return m


def test_features_do_not_peek_at_current_or_future_bars() -> None:
    m = fake_master()
    cutoff = m.hour.iloc[400]
    a = build(m)

    m2 = m.copy()
    mask = m2.hour >= cutoff                      # corrupt EVERYTHING from bar n onwards
    cols = ["close", "volume", "n_tweets", "sent_mean", "w_follow", "w_engage", "pct_pos"]
    m2.loc[mask, cols] *= 7.0
    b = build(m2)

    feats = feature_columns(a)
    fa = a[a.hour <= cutoff].set_index("hour")[feats]
    fb = b[b.hour <= cutoff].set_index("hour")[feats]
    assert fa.shape == fb.shape and len(fa) > 100
    np.testing.assert_allclose(fa.values, fb.values, rtol=1e-9, equal_nan=True)


def test_future_row_has_features_but_no_targets() -> None:
    m = fake_master()
    d = build(m, add_future_row=True)
    last = d.iloc[-1]
    assert last.hour == m.hour.iloc[-1] + pd.Timedelta(hours=1)
    assert last[["y0", "y5"]].isna().all()
    assert last[feature_columns(d)].notna().all()
