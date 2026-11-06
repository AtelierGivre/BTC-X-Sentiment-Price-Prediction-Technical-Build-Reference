# BTC X-Sentiment Price Prediction

A from-scratch Python pipeline that forecasts BTC's price at **n+0 … n+5 hours** using
**X (Twitter) sentiment and price only**, validated with walk-forward testing and reported
with Sharpe, Sortino and Max Drawdown.

```
Apify tweets ─┐
              ├─ clean → sentiment → hourly master → features → walk-forward models ─┐
Binance price ┘                                                                     ├─ metrics + charts + reports
Tavily news ────────────────────────────────────────────────→ 1-page summary ───────┘
                                                              (NOT a model input)
```

## Assessment brief — where each item is met

| Brief item | Where |
|---|---|
| Scrape X posts (ticker / cashtag / keyword) via Apify | `src/ingest/tweets_apify.py`, query `($BTC OR #Bitcoin OR bitcoin)` |
| Historical price for the same period | `src/ingest/prices.py` — Binance 1h klines |
| Timestamps (UTC), post text, likes, reposts, views, follower counts | `tweets_raw.parquet` -> `likes`, `reposts`, `replies`, `views`, `followers` |
| Align posts with price bars | `src/process/aggregate.py` — hourly master, joined in DuckDB (`master_sql`) |
| Recent news via Tavily | `src/ingest/news_tavily.py`, surfaced in the 1-page summary |
| Time-based features from sentiment + price only | `src/features/build.py` — 56 columns, leakage-guarded |
| 1-page current sentiment outlook | `reports/sentiment_summary.md` |
| Model for n+0 … n+5 | `src/models/walkforward.py` — Ridge, LightGBM, naive |
| Walk-forward validation | 4 folds, 6-bar purge, 672 out-of-sample hours |
| Visual matrix output generator | `reports/figures/01…08` — sentiment vs price over time, plus the lag × horizon matrix |
| Evaluation report (Sharpe / Sortino / Max Drawdown) | `reports/evaluation_report.md` §7 |
| Tokens + actor IDs for reproduction | `.env.example`; `.env` is never committed |

## Fixed conventions (these prevent leakage)

| Item | Rule |
|---|---|
| Asset | BTCUSDT |
| Bar | 1 hour, labelled by **open time**, stored in **UTC** |
| Decision time | start of bar `n` |
| Information | features for row `n` use only data up to the **close of bar n-1** |
| Target | `y_h = log(close[n+h] / close[n-1])`, `h = 0..5` |
| Price forecast | `close[n-1] * exp(y_h)` |
| Purge gap | 6 bars between train end and test start |
| Costs | 7 bps per position change |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the keys you have. `.env` is gitignored and is
**never** committed.

### Keys

| Variable | Needed for | Notes |
|---|---|---|
| `APIFY_TOKEN` | tweet scrape | required for a real run |
| `APIFY_ACTOR_ID` | tweet scrape | pick an actor on apify.com/store and adapt `build_run_input()` |
| `TAVILY_API_KEY` | 1-page news summary | optional; the pipeline still runs without it |
| `TRANSFORMER_MODEL` | optional sentiment model | optional; VADER is the baseline |

## Run without keys (synthetic demo)

```bash
python run_pipeline.py --stage all --demo
pytest -q
```

Writes to `data_demo/` and `reports_demo/`. The demo plants a weak known signal so you can
see the pipeline find it. **Demo numbers are not real results.**

## Run for real

1. `python run_pipeline.py --stage ingest` — prices (no key) + Apify tweets.
2. Set `APIFY_ACTOR_ID` in `.env`. The default pipeline is wired for
   `wAusCMrm284Voaw86` (`xquik~x-tweet-scraper`), the cheapest actor tested that
   **honours historical time windows** (~$0.15–0.17 per 1,000 tweets on the FREE tier).
3. Test an **old** window first (historical availability is the real risk):
   `python -m src.ingest.tweets_apify --test --date 2026-09-20`
   Inspect `data/raw/apify_windows_test/` and the logged cost.
4. Run the full scrape: `python -m src.ingest.tweets_apify`
   It is idempotent and stops safely when the Apify credit runs low.
5. `python run_pipeline.py --stage process`
6. `python run_pipeline.py --stage train`
7. `python run_pipeline.py --stage report`

### Budget reality (FREE Apify plan)

The FREE plan includes **$5/month**, so a 180-day scrape is not affordable at
~$0.15/1K tweets. The shipped configuration buys **60 days × 480 tweets/day ≈ 28.8K
tweets for ≈ $4.32**, leaving headroom for the credit check. One `Top`-sorted query
per day covers ~23 of 24 hours; `Latest` sorting was tested and rejected because it
returns only the final minutes of each window. The shorter tweet window is disclosed
in the evaluation report — it is never silently padded with another dataset.

Actor notes:

| Actor | ID | Free-tier price | Verdict |
|---|---|---|---|
| `xquik~x-tweet-scraper` | `wAusCMrm284Voaw86` | ~$0.15/1K | **used**: honours ISO windows |
| `scrape.badger~twitter-tweets-scraper` | `pzMmk1t7AZ8OKJhfU` | $0.15/1K | rejected: ignores `since`/`until` |
| `kaitoeasyapi~...cheapest` | `CJdippxWmn9uRfooo` | $0.25/1K | rejected: returned records without timestamps |
| `apidojo~twitter-scraper-lite` | `nfp1fpt5gUlBwPcor` | demo only | rejected: 5 runs/month × 10 items on FREE |
| `coder_luffy~free-tweet-scraper` | `hfNNTkPlcLkjKJYKj` | $1.99/1K | rejected: needs tweet IDs, no search |

Outputs: `reports/evaluation_report.md`, `reports/sentiment_summary.md`,
`reports/figures/*.html`, `reports/metrics_*.csv`, `logs/pipeline.log`.
Small pre-computed tables are copied to `artifacts/` so the report can be rebuilt offline.

## Folder map

```
config.py                    settings + paths
run_pipeline.py              stage runner
src/ingest/                  prices.py, tweets_apify.py, news_tavily.py, synthetic.py
src/process/                 clean.py -> sentiment.py -> aggregate.py
src/features/build.py        features, targets, leakage guard
src/models/walkforward.py    folds, naive/Ridge/LightGBM, ablations
src/evaluation/              metrics.py, backtest.py
src/visualization/charts.py  8 interactive charts
src/reports/                 summary.py, evaluation_report.py
tests/                       leakage, folds, metrics
```

## Fallbacks (implemented in code, not aspirational)

- **Binance blocked / HTTP 451** → `fetch_prices()` tries `api.binance.com`, then
  `data-api.binance.vision`, then yfinance (`BTC-USD`, hourly history limited to ~730 days).
- **Apify too expensive / flaky** → scrape a shorter real window (e.g. 30 days) and say so in
  the report. Do **not** silently fill the gap with Kaggle data.
- **Transformer problems** → stay on VADER; nothing else depends on it.
- **Tavily down or key missing** → the summary still builds, just without headlines.

