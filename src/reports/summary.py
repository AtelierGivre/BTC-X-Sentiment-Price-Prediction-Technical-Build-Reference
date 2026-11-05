"""Build the ~1-page sentiment outlook. Tavily headlines appear here and nowhere else."""
from __future__ import annotations

import json

import pandas as pd

from config import PROC, RAW, REPORTS, SYMBOL
from src.utils.log import get_logger

log = get_logger(__name__)


def _mean(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].mean()) if len(frame) else float("nan")


def _snippet(news: dict, limit: int = 180) -> str:
    """One-line preview of the article body, without repeating the headline."""
    title = str(news.get("title") or "").strip()
    body = " ".join(str(news.get("content") or "").split())
    if title and body.lower().startswith(title.lower()):
        body = body[len(title):].lstrip(" .:-—")
    return body[:limit]


def main() -> None:
    m = pd.read_parquet(PROC / "master.parquet").sort_values("hour")
    fc = pd.read_csv(PROC / "final_forecast.csv")
    news_path = RAW / "news.json"
    news = json.loads(news_path.read_text(encoding="utf-8")) if news_path.exists() else []

    s24, s7, s30 = m.tail(24), m.tail(168), m.tail(720)
    prior24 = m.tail(48).head(24)
    sent_now, sent_7, sent_30 = _mean(s24, "w_follow"), _mean(s7, "w_follow"), _mean(s30, "w_follow")
    vol_now, vol_7 = _mean(s24, "n_tweets"), _mean(s7, "n_tweets")
    delta = sent_now - _mean(prior24, "w_follow")
    last = m.tail(1).iloc[0]

    fc_rows = "\n".join(
        f"| n+{int(r.h)} | {r.ridge_price:,.0f} | {r.lgbm_price:,.0f} |"
        for r in fc.itertuples()
    )
    headlines = "\n".join(
        f"- **{n.get('title', 'untitled')}** — {n.get('url', '')}  \n  {_snippet(n)}"
        for n in news[:5]
    ) or "- No Tavily headlines (key not set or no results)."

    direction = "rising" if delta > 0.01 else "falling" if delta < -0.01 else "flat"
    text = f"""# Sentiment outlook — {SYMBOL}

*As of bar {last.hour} (UTC). Model output, not investment advice.*

## Current reading
- **Follower-weighted sentiment (last 24h):** {sent_now:+.3f} — 7-day {sent_7:+.3f}, 30-day {sent_30:+.3f}; momentum vs prior 24h: **{delta:+.3f} ({direction})**.
- **Tweet volume:** {vol_now:.0f}/hour last 24h vs {vol_7:.0f}/hour over 7 days.
- **Positive share (24h):** {_mean(s24, 'pct_pos'):.1%}; negative share: {_mean(s24, 'pct_neg'):.1%}.
- **No-tweet hours in the last day:** {int(s24.no_tweets.sum())}.

## Model view (latest bar)
| horizon | Ridge price | LightGBM price |
|---|---|---|
{fc_rows}

## News drivers (Tavily — summary only, not a model input)
{headlines}

## Outlook
Sentiment over the last day is {direction} at {sent_now:+.3f}, with volume at {vol_now:.0f} tweets/hour.
The model's near-term path is shown above; the spread between Ridge and LightGBM is a useful
indicator of disagreement. Any edge here is small and regime-dependent, and the backtest should
be read alongside its bootstrap interval and transaction costs.

*Caveats: short sample, bot noise, scraper sampling limits, and fees with no slippage model.*
"""
    out = REPORTS / "sentiment_summary.md"
    out.write_text(text, encoding="utf-8")
    log.info("sentiment summary written -> %s", out)


if __name__ == "__main__":
    main()
