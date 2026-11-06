from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.process.aggregate import build_master, trim_to_tweets, write_database


def test_trim_keeps_tweet_window_plus_warmup() -> None:
    prices = pd.DataFrame({
        "hour": pd.date_range("2026-01-01", periods=240, freq="h", tz="UTC"),
        "close": 1.0,
    })
    tweets = pd.DataFrame({
        "created_at": pd.to_datetime([
            "2026-01-09 05:30", "2026-01-09 09:00", "2026-01-10 04:00",
        ], utc=True),
    })

    out = trim_to_tweets(prices, tweets, warmup_hours=24)

    assert out.hour.min() == pd.Timestamp("2026-01-08 05:00", tz="UTC")
    assert out.hour.max() == prices.hour.max()
    assert len(out) == 67      # Jan 8 05:00 -> Jan 10 23:00 inclusive (10-day sample)


def test_trim_rejects_empty_tweets() -> None:
    prices = pd.DataFrame({"hour": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")})
    with pytest.raises(ValueError):
        trim_to_tweets(prices, pd.DataFrame({"created_at": []}))


def test_single_tweet_hour_has_zero_dispersion_not_carried_value() -> None:
    prices = pd.DataFrame({
        "hour": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
        "close": [100.0, 101.0, 102.0, 103.0, 104.0],
    })
    tweets = pd.DataFrame({
        "created_at": pd.to_datetime(["2026-01-01 01:30"], utc=True),
        "s_vader": [0.5],
        "followers": [100],
        "likes": [1],
        "reposts": [0],
        "replies": [0],
        "views": [10],
    })

    m = build_master(prices, tweets).set_index("hour")

    # one opinion means zero measured dispersion, not the previous hour's std
    assert m.loc["2026-01-01 01:00", "n_tweets"] == 1
    assert m.loc["2026-01-01 01:00", "sent_std"] == 0.0
    assert m.loc["2026-01-01 01:00", "no_tweets"] == 0.0


def test_sql_join_matches_pandas_master(tmp_path) -> None:
    """The DuckDB view must reproduce the same hourly join the pandas code builds."""
    prices = pd.DataFrame({
        "hour": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
        "open": [1.0] * 6, "high": [1.0] * 6, "low": [1.0] * 6,
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0], "volume": [10.0] * 6,
    })
    tweets = pd.DataFrame({
        "created_at": pd.to_datetime(
            ["2026-01-01 01:30", "2026-01-01 01:45", "2026-01-01 03:10"], utc=True
        ),
        "s_vader": [0.4, 0.6, -0.2],
        "followers": [100, 300, 50],
        "likes": [1, 4, 0],
        "reposts": [0, 1, 0],
        "replies": [0, 0, 0],
        "views": [10, 20, 5],
    })

    master = build_master(prices, tweets)
    db = tmp_path / "check.duckdb"
    write_database(prices, tweets, master, path=db)

    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    sql = con.execute("SELECT * FROM master_sql ORDER BY hour").df()
    con.close()

    assert len(sql) == len(master) == 6
    sql = sql.merge(master, on="hour", suffixes=("_sql", "_pd"))
    for col in ("n_tweets", "no_tweets", "sent_mean", "sent_median", "sent_std",
                "pct_pos", "pct_neg", "w_follow", "w_engage", "views_sum", "engage_sum"):
        assert np.allclose(sql[f"{col}_sql"], sql[f"{col}_pd"], atol=1e-9), col
