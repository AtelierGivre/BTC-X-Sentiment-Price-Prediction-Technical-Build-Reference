"""Copy the small processed tables into `artifacts/` for offline reproduction.

The raw scrape is ~196 MB and the Apify credit is spent, so the full `data/` tree stays
gitignored. These six tables (~1.2 MB) are enough to rebuild every number in the report
without touching Apify. `scrape_summary.json` carries the real coverage and cost so the
report's data section can be rebuilt without the raw tweets.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from config import (
    APIFY_ACTOR_ID, MAX_ITEMS_PER_WINDOW, PROC, RAW, ROOT, SEARCH_QUERY, WINDOW_HOURS,
)
from src.utils.log import get_logger

log = get_logger(__name__)

FILES = [
    "master.parquet",
    "features.parquet",
    "predictions.parquet",
    "strategy_returns.parquet",
    "feature_importance.parquet",
    "final_forecast.csv",
]


def _scrape_summary() -> dict | None:
    """Coverage/cost of the real scrape, or None when only artifacts are present."""
    tweets_path = RAW / "tweets_raw.parquet"
    if not tweets_path.exists():
        return None
    tweets = pd.read_parquet(tweets_path, columns=["created_at"])
    metas = list((RAW / "apify_windows").glob("w_*.meta.json"))
    cost = 0.0
    for f in metas:
        try:
            cost += float(json.loads(f.read_text(encoding="utf-8")).get("cost_usd") or 0)
        except Exception:
            pass
    start, end = tweets.created_at.min(), tweets.created_at.max()
    return {
        "n_tweets": int(len(tweets)),
        "tweet_start": start.isoformat(),
        "tweet_end": end.isoformat(),
        "tweet_days": round((end - start).total_seconds() / 86400, 6),
        "cost_usd": round(cost, 6),
        "n_windows": len(metas),
        "actor_id": APIFY_ACTOR_ID,
        "query": SEARCH_QUERY,
        "sort": "Top",
        "max_items_per_window": MAX_ITEMS_PER_WINDOW,
        "window_hours": WINDOW_HOURS,
    }


def main() -> None:
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    rows = []
    for name in FILES:
        src = PROC / name
        if not src.exists():
            log.warning("snapshot | missing %s", name)
            continue
        shutil.copy2(src, out / name)
        if src.suffix == ".parquet":
            n = len(pd.read_parquet(src))
        else:
            n = max(0, sum(1 for _ in src.open(encoding="utf-8")) - 1)
        rows.append(f"| `{name}` | {n:,} | {(out / name).stat().st_size / 1024:.0f} KB |")

    summary = _scrape_summary()
    if summary:
        (out / "scrape_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    news = RAW / "news.json"
    if news.exists():
        shutil.copy2(news, out / "news.json")

    text = (
        "# Pre-computed outputs\n\n"
        "Small processed tables from the real run, committed so the evaluation report can be\n"
        "reproduced without re-scraping X (the raw scrape is ~196 MB and stays out of git).\n\n"
        "| file | rows | size |\n|---|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n`scrape_summary.json` records the real scrape coverage and cost, so §2 of the\n"
        "report can be rebuilt without the raw tweets. `news.json` preserves the Tavily\n"
        "headlines used by the 1-page sentiment summary.\n\n"
        "Restore the tables with `python scripts/restore_artifacts.py`, then run\n"
        "`python run_pipeline.py --stage report` (or `--stage train` first to refit the\n"
        "models). See `REPRODUCE.md` for the full instructions.\n"
    )
    (out / "README.md").write_text(text, encoding="utf-8")
    log.info("snapshot: %d tables -> %s", len(rows), out)


if __name__ == "__main__":
    main()
