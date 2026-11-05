"""Clean raw tweets and log the row count after every step.

Input : data/raw/tweets_raw.parquet
Output: data/interim/tweets_clean.parquet + reports/cleaning_log.json
"""
from __future__ import annotations

import json
import re

import pandas as pd

from config import INTERIM, RAW, REPORTS
from src.utils.log import get_logger

log = get_logger(__name__)

SPAM = re.compile(
    r"(?:giveaway|airdrop|dm me|t\.me/|join (?:our|my) (?:telegram|discord)|free (?:btc|bitcoin))",
    re.I,
)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts: dict[str, int] = {"start": len(df)}
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df[df.created_at.notna()]

    # 1) duplicate ids
    df = df.drop_duplicates("tweet_id")
    counts["after_dedup_tweet_id"] = len(df)

    # 2) strip URLs / @mentions / extra whitespace (keep the original in `text`)
    df["text_clean"] = (
        df.text.str.replace(r"https?://\S+|www\.\S+", "", regex=True)
        .str.replace(r"@\w+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df.text_clean.str.len() > 5]
    counts["after_nonempty"] = len(df)

    # 3) cheap English filter: drop text that is mostly non-ASCII
    non_ascii = df.text_clean.map(lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1))
    df = df[non_ascii <= 0.30]
    counts["after_english_filter"] = len(df)

    df["hour"] = df.created_at.dt.floor("h")
    df["day"] = df.created_at.dt.floor("D")
    df["key"] = df.text_clean.str.lower()

    # 4) same author repeating the same text within an hour
    df = df.drop_duplicates(["author_id", "hour", "key"])
    counts["after_author_repeats"] = len(df)

    # 5) bot swarms: identical text posted >= 10 times in a day -> keep only the first
    n_same = df.groupby(["day", "key"])["tweet_id"].transform("size")
    df = df[~((n_same >= 10) & df.duplicated(["day", "key"]))]
    counts["after_swarm_dedup"] = len(df)

    # 6) obvious spam from tiny accounts
    spam = df.text.str.contains(SPAM) & (df.followers < 10)
    df = df[~spam]
    counts["after_spam_filter"] = len(df)

    df = df.drop(columns=["day", "key"]).sort_values("created_at").reset_index(drop=True)
    return df, counts


def main() -> None:
    raw = pd.read_parquet(RAW / "tweets_raw.parquet")
    out, counts = clean(raw)
    out.to_parquet(INTERIM / "tweets_clean.parquet", index=False)
    (REPORTS / "cleaning_log.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    for key, value in counts.items():
        log.info("clean | %-24s %d", key, value)


if __name__ == "__main__":
    main()
