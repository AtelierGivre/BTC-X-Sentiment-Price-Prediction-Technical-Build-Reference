# Sentiment outlook — BTCUSDT

*As of bar 2026-11-05 05:00:00+00:00 (UTC). Model output, not investment advice.*

## Current reading
- **Follower-weighted sentiment (last 24h):** +0.223 — 7-day +0.167, 30-day +0.145; momentum vs prior 24h: **+0.023 (rising)**.
- **Tweet volume:** 20/hour last 24h vs 20/hour over 7 days.
- **Positive share (24h):** 57.6%; negative share: 23.4%.
- **No-tweet hours in the last day:** 0.

## Model view (latest bar)
| horizon | Ridge price | LightGBM price |
|---|---|---|
| n+0 | 80,408 | 80,459 |
| n+1 | 80,394 | 80,512 |
| n+2 | 80,390 | 80,478 |
| n+3 | 80,399 | 80,497 |
| n+4 | 80,415 | 80,682 |
| n+5 | 80,431 | 80,647 |

## News drivers (Tavily — summary only, not a model input)
- No Tavily headlines (key not set or no results).

## Outlook
Sentiment over the last day is rising at +0.223, with volume at 20 tweets/hour.
The model's near-term path is shown above; the spread between Ridge and LightGBM is a useful
indicator of disagreement. Any edge here is small and regime-dependent, and the backtest should
be read alongside its bootstrap interval and transaction costs.

*Caveats: short sample, bot noise, scraper sampling limits, and fees with no slippage model.*
