"""Score every tweet.

  s_vader : VADER compound score in [-1, 1]. Fast, no model download (the baseline).
  s_tx    : optional transformer (set TRANSFORMER_MODEL in .env). Scores at most
            TX_MAX_PER_HOUR random tweets per hour and caches results, so a crash loses nothing.

Input : data/interim/tweets_clean.parquet
Output: data/interim/tweets_scored.parquet
"""
from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import INTERIM, SEED, TRANSFORMER_MODEL, TX_MAX_PER_HOUR
from src.utils.log import get_logger

log = get_logger(__name__)

# CHECK your model card! Label names and order vary between models.
POS_LABELS = {"positive", "bullish", "pos", "label_1", "label_2"}
NEG_LABELS = {"negative", "bearish", "neg", "label_0"}


def score_vader(df: pd.DataFrame) -> pd.Series:
    sia = SentimentIntensityAnalyzer()
    return df.text_clean.map(lambda t: sia.polarity_scores(t)["compound"])


def score_transformer(df: pd.DataFrame) -> pd.Series:
    """score = P(positive) - P(negative). Verify the labels on 20 tweets before trusting it."""
    from transformers import pipeline

    cache_path = INTERIM / "tx_cache.parquet"
    cache = (
        pd.read_parquet(cache_path)
        if cache_path.exists()
        else pd.DataFrame(columns=["tweet_id", "s_tx"])
    )
    sample = df.sample(frac=1, random_state=SEED).groupby("hour").head(TX_MAX_PER_HOUR)
    todo = sample[~sample.tweet_id.isin(cache.tweet_id)]
    log.info(
        "transformer: %d sampled, %d cached, %d to score",
        len(sample), len(sample) - len(todo), len(todo),
    )
    if len(todo):
        clf = pipeline(
            "text-classification", model=TRANSFORMER_MODEL, top_k=None, device=-1, truncation=True
        )
        chunk = 2000
        for i in range(0, len(todo), chunk):
            part = todo.iloc[i:i + chunk]
            outs = clf(part.text_clean.tolist(), batch_size=32, truncation=True, max_length=64)
            scores = [
                sum(d["score"] for d in out if d["label"].lower() in POS_LABELS)
                - sum(d["score"] for d in out if d["label"].lower() in NEG_LABELS)
                for out in outs
            ]
            cache = pd.concat([
                cache,
                pd.DataFrame({"tweet_id": part.tweet_id.values, "s_tx": scores}),
            ], ignore_index=True)
            cache.to_parquet(cache_path, index=False)      # checkpoint after every chunk
            log.info("transformer progress: %d/%d", min(i + chunk, len(todo)), len(todo))
    return df.tweet_id.map(cache.drop_duplicates("tweet_id").set_index("tweet_id")["s_tx"])


def main() -> None:
    df = pd.read_parquet(INTERIM / "tweets_clean.parquet")
    df["hour"] = df.created_at.dt.floor("h")
    df["s_vader"] = score_vader(df)
    if TRANSFORMER_MODEL:
        df["s_tx"] = score_transformer(df)
        corr = df[["s_vader", "s_tx"]].corr().iloc[0, 1]
        log.info("corr(vader, transformer) = %.3f", corr)
    df.drop(columns="hour").to_parquet(INTERIM / "tweets_scored.parquet", index=False)
    log.info("scored %d tweets | mean VADER = %.3f", len(df), df.s_vader.mean())


if __name__ == "__main__":
    main()
