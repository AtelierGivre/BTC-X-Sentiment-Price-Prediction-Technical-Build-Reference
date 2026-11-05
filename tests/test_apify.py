"""Offline tests for the Apify adapter (no network, no token needed)."""
from __future__ import annotations

import json

import pandas as pd

from src.ingest import tweets_apify as t


def test_xquik_input_uses_iso_window_and_top_sort(monkeypatch) -> None:
    monkeypatch.setattr(t, "APIFY_ACTOR_ID", t.XQUIK)
    start = pd.Timestamp("2026-09-20 00:00", tz="UTC").to_pydatetime()
    end = pd.Timestamp("2026-09-21 00:00", tz="UTC").to_pydatetime()

    inp = t.build_run_input(start, end, 480)

    assert inp["mode"] == "search"
    assert inp["queryType"] == "Top"          # "Latest" clusters at end-of-window
    assert inp["maxItems"] == 480
    assert inp["since"] == "2026-09-20T00:00:00Z"
    assert inp["until"] == "2026-09-21T00:00:00Z"
    assert "#Bitcoin" in inp["searchTerms"][0]


def test_normalize_maps_xquik_fields() -> None:
    item = {
        "id": "2101822968405860789",
        "text": "bitcoin looks strong",
        "createdAt": "Sun Sep 20 23:57:00 +0000 2026",
        "likeCount": 277,
        "retweetCount": 26,
        "replyCount": 22,
        "viewCount": 45976,
        "author": {"id": "619075565", "followers": 767, "username": "ben"},
    }
    row = t.normalize(item)
    assert row["tweet_id"] == "2101822968405860789"
    assert row["likes"] == 277
    assert row["reposts"] == 26
    assert row["replies"] == 22
    assert row["views"] == 45976
    assert row["followers"] == 767
    assert row["author_id"] == "619075565"


def test_twitter_date_format_is_parsed() -> None:
    parsed = t._parse_dates(pd.Series(["Sun Sep 20 23:57:00 +0000 2026"]))
    assert parsed.iloc[0] == pd.Timestamp("2026-09-20 23:57:00", tz="UTC")


def test_build_raw_table_skips_meta_and_dedupes(tmp_path, monkeypatch) -> None:
    win = tmp_path / "windows"
    win.mkdir()
    items = [
        {"id": "2", "text": "second", "createdAt": "Sun Sep 20 23:57:00 +0000 2026"},
        {"id": "1", "text": "first", "createdAt": "Sun Sep 20 01:57:00 +0000 2026"},
        {"id": "2", "text": "second again", "createdAt": "Sun Sep 20 23:57:00 +0000 2026"},
    ]
    (win / "w_20260920_00.json").write_text(json.dumps(items), encoding="utf-8")
    (win / "w_20260920_00.meta.json").write_text(json.dumps({"items": 3}), encoding="utf-8")

    monkeypatch.setattr(t, "WIN_DIR", win)
    monkeypatch.setattr(t, "RAW", tmp_path)

    df = t.build_raw_table()

    assert len(df) == 2                      # duplicate id collapsed, meta ignored
    assert list(df.tweet_id) == ["1", "2"]   # chronological
    assert (tmp_path / "tweets_raw.parquet").exists()
