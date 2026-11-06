"""Scrape X/Twitter posts with Apify, one day-sized window per run.

Configured for **xquik~x-tweet-scraper** (actor id wAusCMrm284Voaw86), which charges
about $0.15 per 1,000 tweets on Apify's FREE tier. The FREE plan gives $5/month,
so the scrape is budget-limited (see TWEET_DAYS / APIFY_BUDGET_USD in .env).

Why day windows: X advanced search filters by calendar date (`since:` / `until:`),
so one run per day is the reliable unit. Each tweet still carries a timestamp, so the
pipeline bins to hours downstream. The `until:` date is exclusive.

Usage:
    python -m src.ingest.tweets_apify --test --date 2026-09-20   # ~20 items, costs cents
    python -m src.ingest.tweets_apify --max-windows 2            # smoke test of the real run
    python -m src.ingest.tweets_apify                            # full TWEET_DAYS
    python -m src.ingest.tweets_apify --build-only               # rebuild parquet from saved JSON

Idempotent: finished windows are skipped, so a crash never re-pays for finished work.
Every run's cost and item count are stored in a sidecar `.meta.json`.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from config import (
    APIFY_ACTOR_ID, APIFY_BUDGET_USD, APIFY_TOKEN, MAX_ITEMS_PER_WINDOW, RAW,
    SEARCH_QUERY, TWEET_DAYS, WINDOW_HOURS,
)
from src.utils.log import get_logger

log = get_logger(__name__)
WIN_DIR = RAW / "apify_windows"
TEST_DIR = RAW / "apify_windows_test"
WIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

BADGER = "pzMmk1t7AZ8OKJhfU"          # $0.15/1K tweets, but ignores date bounds
XQUIK = "wAusCMrm284Voaw86"           # $0.15/1K tweets, honours ISO time windows
TWITTER_DATE = "%a %b %d %H:%M:%S %z %Y"   # "Fri Nov 24 17:49:36 +0000 2023"


# --------------------------------------------------------------------------- Apify
def build_run_input(start: datetime, end: datetime, max_items: int) -> dict:
    """Actor-specific input. Add a branch here if you change APIFY_ACTOR_ID."""
    if APIFY_ACTOR_ID == XQUIK:
        # ISO timestamps are honoured to the minute. "Top" spreads the sample across
        # the whole window; "Latest" returns only the final minutes of it.
        return {
            "mode": "search",
            "searchTerms": [SEARCH_QUERY],
            "maxItems": max_items,
            "queryType": "Top",
            "since": f"{start:%Y-%m-%dT%H:%M:%SZ}",
            "until": f"{end:%Y-%m-%dT%H:%M:%SZ}",
        }
    if APIFY_ACTOR_ID == BADGER:
        # NOTE: this actor was measured to IGNORE since/until bounds. Kept only as a fallback.
        query = f"{SEARCH_QUERY} since:{start:%Y-%m-%d} until:{end:%Y-%m-%d}"
        return {
            "mode": "Advanced Search",
            "query": query,
            "query_type": "Latest",
            "max_results": max_items,
        }
    # generic fallback for searchTerms-style actors
    query = (
        f"{SEARCH_QUERY} "
        f"since:{start:%Y-%m-%d_%H:%M:%S}_UTC until:{end:%Y-%m-%d_%H:%M:%S}_UTC"
    )
    return {"searchTerms": [query], "maxItems": max_items, "sort": "Latest"}


def _spent(directory) -> float:
    """Sum finalised cost from window sidecars (real windows + test windows)."""
    total = 0.0
    for d in {Path(directory), TEST_DIR}:          # set: calling with TEST_DIR must not double count
        for f in d.glob("w_*.meta.json"):
            try:
                total += float(json.loads(f.read_text(encoding="utf-8")).get("cost_usd") or 0)
            except Exception:
                pass
    return total


class BudgetStop(RuntimeError):
    pass


def account_remaining() -> float | None:
    """Remaining USD credit in the current Apify billing cycle, or None if unknown."""
    import requests

    try:
        r = requests.get(
            "https://api.apify.com/v2/users/me/limits",
            params={"token": APIFY_TOKEN}, timeout=30,
        )
        r.raise_for_status()
        d = r.json()["data"]
        cap = float(d["limits"].get("maxMonthlyUsageUsd", 5.0))
        used = float(d["current"].get("monthlyUsageUsd", 0.0))
        return cap - used
    except Exception as exc:
        log.warning("could not read Apify credit: %s", exc)
        return None


def scrape_window(client, start: datetime, end: datetime, max_items: int,
                  out_dir=WIN_DIR) -> float:
    """Run one window, save items + cost metadata. Returns the cost in USD."""
    out = out_dir / f"w_{start:%Y%m%d_%H}.json"
    meta_path = out_dir / f"w_{start:%Y%m%d_%H}.meta.json"
    if out.exists():
        log.info("window %s already saved, skipping", start)
        return 0.0

    spent = _spent(WIN_DIR)
    est = max_items * 0.00015
    remaining = account_remaining()
    if remaining is not None:
        log.info("Apify credit remaining: $%.2f | this window up to $%.3f", remaining, est)
        if remaining < est + 0.15:
            raise BudgetStop(f"only ${remaining:.2f} credit left; window needs ${est:.2f}")
    elif spent + est > APIFY_BUDGET_USD:
        raise BudgetStop(
            f"spent ${spent:.2f}, next window up to ${est:.2f}, "
            f"cap ${APIFY_BUDGET_USD:.2f}"
        )

    # hard per-run cap, so a bad actor cannot blow the monthly credit
    cap = Decimal(str(round(min(max_items * 0.00015 * 1.5, 0.25), 4)))
    run = client.actor(APIFY_ACTOR_ID).call(
        run_input=build_run_input(start, end, max_items),
        run_timeout=timedelta(minutes=30),
        max_total_charge_usd=cap,
        logger=None,
    )
    if run is None:
        raise RuntimeError("Apify returned no run object")
    items = list(client.dataset(run.default_dataset_id).iterate_items())
    out.write_text(json.dumps(items), encoding="utf-8")

    # usage_total_usd on the object returned by .call() can be stale; re-fetch once.
    try:
        run = client.run(run.id).get() or run
    except Exception:
        pass
    cost = max(float(run.usage_total_usd or 0), len(items) * 0.00015)

    meta_path.write_text(json.dumps({
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "items": len(items),
        "cost_usd": cost,
        "run_id": run.id,
        "status": getattr(run, "status", None),
        "spent_after_usd": round(spent + cost, 4),
    }, indent=2), encoding="utf-8")
    log.info("window %s -> %d items | cost=$%.4f | spent=$%.4f",
             start, len(items), cost, spent + cost)
    return cost


# ---------------------------------------------------------------------- normalise
def _get(d: dict, *paths):
    """Return the first non-None value found at any of the dotted key paths."""
    for path in paths:
        cur = d
        for key in path.split("."):
            cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is not None:
            return cur
    return None


def normalize(item: dict) -> dict:
    """Map actor-specific field names to ONE schema."""
    return {
        "tweet_id": str(_get(item, "id", "id_str", "tweetId", "rest_id") or ""),
        "created_at": _get(item, "createdAt", "created_at", "timestamp", "date"),
        "text": _get(item, "text", "fullText", "full_text") or "",
        "likes": _get(item, "likeCount", "favoriteCount", "favorite_count", "likes") or 0,
        "reposts": _get(item, "retweetCount", "retweet_count", "repostCount", "reposts") or 0,
        "replies": _get(item, "replyCount", "reply_count", "replies") or 0,
        "views": _get(item, "viewCount", "views", "impressionCount") or 0,
        "followers": _get(
            item, "author.followers", "author.followersCount", "author.followers_count",
            "user.followers_count", "user.followersCount",
        ) or 0,
        "author_id": str(_get(
            item, "author.id", "author.userName", "user.id_str",
            "author.screen_name", "user.screen_name",
        ) or ""),
    }


def _parse_dates(s: pd.Series) -> pd.Series:
    """X's 'Fri Nov 24 17:49:36 +0000 2023' format first, then ISO as a fallback."""
    out = pd.to_datetime(s, format=TWITTER_DATE, utc=True, errors="coerce")
    missing = out.isna()
    if missing.any():
        out.loc[missing] = pd.to_datetime(s[missing], utc=True, errors="coerce")
    return out


def build_raw_table() -> pd.DataFrame:
    files = sorted(f for f in WIN_DIR.glob("w_*.json") if not f.name.endswith(".meta.json"))
    if not files:
        raise SystemExit(f"No windows found in {WIN_DIR}. Run a scrape first.")
    rows: list[dict] = []
    for f in files:
        rows += [normalize(i) for i in json.loads(f.read_text(encoding="utf-8"))]

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Scrape files are empty; check the actor input and date range.")
    df["created_at"] = _parse_dates(df["created_at"])
    df = df[(df.tweet_id != "") & df.created_at.notna() & (df.text != "")].copy()
    # Adjacent windows can return the same tweet at a boundary or if it is pinned.
    df = df.drop_duplicates("tweet_id").sort_values("created_at").reset_index(drop=True)
    for col in ["likes", "reposts", "replies", "views", "followers"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["source"] = "apify"
    df.to_parquet(RAW / "tweets_raw.parquet", index=False)

    cost = _spent(WIN_DIR)
    span = (df.created_at.max() - df.created_at.min()) if len(df) else pd.Timedelta(0)
    log.info(
        "tweets_raw saved: %d rows | %s -> %s (%.1f days) | total cost $%.2f",
        len(df), df.created_at.min(), df.created_at.max(), span.total_seconds() / 86400, cost,
    )
    return df


# ---------------------------------------------------------------------- orchestrate
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="one small window, 20 items")
    ap.add_argument("--date", help="window start date for --test (YYYY-MM-DD)")
    ap.add_argument("--build-only", action="store_true", help="rebuild parquet from saved JSON")
    ap.add_argument("--max-windows", type=int, default=None, help="stop after N new windows")
    args = ap.parse_args(argv)

    if not args.build_only:
        if not (APIFY_TOKEN and APIFY_ACTOR_ID):
            raise SystemExit("Set APIFY_TOKEN and APIFY_ACTOR_ID in .env first.")
        from apify_client import ApifyClient

        client = ApifyClient(APIFY_TOKEN)
        step = timedelta(hours=WINDOW_HOURS)

        if args.test:
            if args.date:
                start = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                start = datetime.now(timezone.utc).replace(
                    minute=0, second=0, microsecond=0
                ) - timedelta(days=TWEET_DAYS - 1)
            scrape_window(client, start, start + step, 20, out_dir=TEST_DIR)
            return                                   # do not pollute the real dataset

        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(days=TWEET_DAYS)
        windows = []
        t = start
        while t + step <= now:
            windows.append(t)
            t += step

        # Newest first: if the budget stops us early we keep a recent, contiguous
        # block instead of an old one.
        windows.reverse()
        done = 0
        for t in windows:
            existed = (WIN_DIR / f"w_{t:%Y%m%d_%H}.json").exists()
            try:
                scrape_window(client, t, t + step, MAX_ITEMS_PER_WINDOW)
            except BudgetStop as exc:
                log.warning("budget stop: %s", exc)
                break
            if not existed:
                done += 1
                if args.max_windows is not None and done >= args.max_windows:
                    log.info("stopping after %d new windows", done)
                    break

    build_raw_table()


if __name__ == "__main__":
    main()
