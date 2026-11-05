"""DEMO MODE ONLY: synthetic prices + tweets so the whole pipeline runs without keys.

A weak, known signal is planted on purpose: a hidden 'mood' drives tweet wording and,
slightly, the NEXT hour's return. A working pipeline should find a small positive IC.
Never report demo numbers as real results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DAYS, RAW, SEED
from src.utils.log import get_logger

log = get_logger(__name__)

POS = [
    "bullish breakout incoming", "love this rally great news", "btc to the moon amazing gains",
    "strong buy signal looking good", "adoption growing excellent",
]
NEG = [
    "bearish dump coming", "terrible crash awful news", "btc rekt panic selling",
    "weak sell signal looking bad", "fear everywhere horrible",
]
NEU = ["btc price update today", "bitcoin trading volume report", "watching the chart this hour"]
VOCAB = [f"w{i}" for i in range(60)]


def make_demo(days: int = DAYS, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").floor("h")
    hours = pd.date_range(end - pd.Timedelta(days=days), end, freq="h", inclusive="left")
    n = len(hours)

    mood = np.zeros(n)
    for t in range(1, n):
        mood[t] = 0.9 * mood[t - 1] + rng.normal(0, 0.3)

    sigma = 0.004
    ret = rng.normal(0, sigma, n)
    ret[1:] += 0.25 * sigma * np.tanh(mood[:-1])          # mood(t-1) -> return(t)
    close = 60000 * np.exp(np.cumsum(ret))
    prev = np.r_[close[0], close[:-1]]
    prices = pd.DataFrame({
        "hour": hours,
        "open": prev,
        "high": np.maximum(prev, close) * 1.001,
        "low": np.minimum(prev, close) * 0.999,
        "close": close,
        "volume": rng.lognormal(5, 0.4, n),
    })

    recs, tid = [], 0
    for t in range(n):
        k = rng.poisson(15 * (1 + 0.4 * np.sin(2 * np.pi * hours[t].hour / 24)))
        p_pos = 0.5 + 0.35 * np.tanh(mood[t])
        p_neg = 0.5 - 0.35 * np.tanh(mood[t])
        for _ in range(k):
            u = rng.random()
            base = POS if u < p_pos * 0.6 else NEG if u > 1 - p_neg * 0.6 else NEU
            filler = " ".join(rng.choice(VOCAB, 3))
            spam = rng.random() < 0.03
            text = "FREE BTC giveaway DM me " + filler if spam else f"{rng.choice(base)} {filler}"
            tid += 1
            recs.append((
                f"t{tid}",
                hours[t] + pd.Timedelta(seconds=int(rng.integers(0, 3600))),
                text,
                int(rng.lognormal(1, 1.5)),
                int(rng.lognormal(0.5, 1.2)),
                int(rng.poisson(1)),
                int(rng.lognormal(6, 1.5)),
                int(rng.integers(0, 5) if spam else rng.lognormal(6, 2)),
                f"a{rng.integers(0, 5000)}",
            ))

    tweets = pd.DataFrame(recs, columns=[
        "tweet_id", "created_at", "text", "likes", "reposts", "replies",
        "views", "followers", "author_id",
    ])
    # duplicate ids on purpose so the cleaning log has something to show
    tweets = pd.concat([tweets, tweets.sample(frac=0.02, random_state=seed)], ignore_index=True)
    tweets["source"] = "synthetic"
    return prices, tweets


def main() -> None:
    prices, tweets = make_demo()
    prices.to_parquet(RAW / "prices_1h.parquet", index=False)
    tweets.to_parquet(RAW / "tweets_raw.parquet", index=False)
    log.info("DEMO data written: %d bars, %d tweets", len(prices), len(tweets))


if __name__ == "__main__":
    main()
