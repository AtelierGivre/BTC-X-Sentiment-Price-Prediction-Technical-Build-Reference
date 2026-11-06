"""Squash tweets into ONE ROW PER HOUR and join with prices -> the master table.

A row labelled hour H describes tweets posted during [H, H+1h) and the price bar opened at H.
That information is only fully known at the END of hour H, so the feature builder later shifts
by one bar. Hours with no tweets get a flag and a limited forward-fill.

Input : data/interim/tweets_scored.parquet, data/raw/prices_1h.parquet
Output: data/processed/master.parquet (+ DuckDB table `master`)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import INTERIM, POS_THRESH, PROC, RAW, SENT_COL
from src.utils.log import get_logger

log = get_logger(__name__)

SENT_COLS = ["sent_mean", "sent_median", "sent_std", "pct_pos", "pct_neg", "w_follow", "w_engage"]


def hourly_sentiment(tweets: pd.DataFrame, col: str) -> pd.DataFrame:
    tw = tweets[tweets[col].notna()].copy()
    tw["hour"] = tw.created_at.dt.floor("h")
    tw["s"] = tw[col].astype(float)
    tw["w1"] = 1 + np.log1p(tw.followers.clip(lower=0))          # follower weight
    tw["w2"] = 1 + np.log1p((tw.likes + tw.reposts).clip(lower=0))  # engagement weight
    tw["sw1"] = tw.s * tw.w1
    tw["sw2"] = tw.s * tw.w2
    tw["pos"] = (tw.s > POS_THRESH).astype(float)
    tw["neg"] = (tw.s < -POS_THRESH).astype(float)

    g = tw.groupby("hour")
    return pd.DataFrame({
        "n_tweets": g.size(),
        "sent_mean": g.s.mean(),
        "sent_median": g.s.median(),
        "sent_std": g.s.std().fillna(0.0),
        "pct_pos": g.pos.mean(),
        "pct_neg": g.neg.mean(),
        "w_follow": g.sw1.sum() / g.w1.sum(),
        "w_engage": g.sw2.sum() / g.w2.sum(),
        "views_sum": g.views.sum(),
        "engage_sum": g.apply(lambda x: (x.likes + x.reposts + x.replies).sum()),
    })


def build_master(prices: pd.DataFrame, tweets: pd.DataFrame, col: str = SENT_COL) -> pd.DataFrame:
    hourly = hourly_sentiment(tweets, col)
    m = prices.set_index("hour").join(hourly, how="left")
    m["n_tweets"] = m["n_tweets"].fillna(0)
    m["no_tweets"] = (m["n_tweets"] == 0).astype(float)

    # carry sentiment forward for at most 3 empty hours, then neutral 0
    m[SENT_COLS] = m[SENT_COLS].ffill(limit=3).fillna(0.0)
    for c in ("views_sum", "engage_sum"):
        m[c] = m[c].fillna(0.0)
    return m.reset_index()


def trim_to_tweets(prices: pd.DataFrame, tweets: pd.DataFrame, warmup_hours: int = 24) -> pd.DataFrame:
    """Keep only prices from the tweet window (plus warm-up).

    Prices cover 180 days (free), tweets cover only what the Apify budget allowed.
    Training on months where sentiment is a fake 0 would be worse than a shorter sample.
    """
    if tweets.empty:
        raise ValueError("no tweets to align with; run the scrape first")
    start = tweets.created_at.min().floor("h") - pd.Timedelta(hours=warmup_hours)
    return prices[prices.hour >= start].reset_index(drop=True)


def write_database(prices: pd.DataFrame, tweets: pd.DataFrame, master: pd.DataFrame,
                   path=None) -> None:
    """Persist both sources and join them IN SQL (the brief asks for a database join).

    Tables: prices, tweets_hourly, master.  View: master_sql = prices LEFT JOIN
    tweets_hourly USING (hour), i.e. the same join the pandas code performs, executed by
    DuckDB so the alignment can be checked from the database itself.
    """
    import duckdb

    path = path or PROC / "btc.duckdb"
    hourly = hourly_sentiment(tweets, SENT_COL).reset_index()
    con = duckdb.connect(str(path))
    con.register("prices_df", prices)
    con.register("hourly_df", hourly)
    con.register("master_df", master)
    con.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM prices_df")
    con.execute("CREATE OR REPLACE TABLE tweets_hourly AS SELECT * FROM hourly_df")
    con.execute("CREATE OR REPLACE TABLE master AS SELECT * FROM master_df")
    con.execute("DROP VIEW IF EXISTS master_sql")
    con.execute(
        """
        CREATE VIEW master_sql AS
        WITH joined AS (
            SELECT p.hour, p.open, p.high, p.low, p.close, p.volume,
                   h.n_tweets, h.sent_mean, h.sent_median, h.sent_std, h.pct_pos,
                   h.pct_neg, h.w_follow, h.w_engage, h.views_sum, h.engage_sum
            FROM prices p
            LEFT JOIN tweets_hourly h USING (hour)
        )
        SELECT hour, open, high, low, close, volume,
               COALESCE(n_tweets, 0) AS n_tweets,
               COALESCE(last_value(sent_mean   IGNORE NULLS) OVER w, 0.0) AS sent_mean,
               COALESCE(last_value(sent_median IGNORE NULLS) OVER w, 0.0) AS sent_median,
               COALESCE(last_value(sent_std    IGNORE NULLS) OVER w, 0.0) AS sent_std,
               COALESCE(last_value(pct_pos     IGNORE NULLS) OVER w, 0.0) AS pct_pos,
               COALESCE(last_value(pct_neg     IGNORE NULLS) OVER w, 0.0) AS pct_neg,
               COALESCE(last_value(w_follow    IGNORE NULLS) OVER w, 0.0) AS w_follow,
               COALESCE(last_value(w_engage    IGNORE NULLS) OVER w, 0.0) AS w_engage,
               COALESCE(views_sum, 0.0) AS views_sum,
               COALESCE(engage_sum, 0.0) AS engage_sum,
               CASE WHEN COALESCE(n_tweets, 0) = 0 THEN 1.0 ELSE 0.0 END AS no_tweets
        FROM joined
        WINDOW w AS (ORDER BY hour ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
        """
    )
    con.close()


def main() -> None:
    prices = pd.read_parquet(RAW / "prices_1h.parquet")
    tweets = pd.read_parquet(INTERIM / "tweets_scored.parquet")

    before = len(prices)
    prices = trim_to_tweets(prices, tweets)
    log.info(
        "master trimmed to tweet coverage: %d -> %d price bars (from %s)",
        before, len(prices), prices.hour.min(),
    )

    master = build_master(prices, tweets)
    assert master.hour.is_monotonic_increasing and master.hour.is_unique
    master.to_parquet(PROC / "master.parquet", index=False)
    write_database(prices, tweets, master)

    log.info(
        "master: %d rows | no-tweet hours: %d | median tweets/hour: %.0f",
        len(master), int(master.no_tweets.sum()), master.n_tweets.median(),
    )


if __name__ == "__main__":
    main()
