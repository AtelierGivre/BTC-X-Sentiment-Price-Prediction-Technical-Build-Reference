# Reproduce the results

Everything needed to rebuild the evaluation is in this repository. Choose the path that
matches how much you want to re-run.

## What is committed

- **Code**: ingestion, cleaning, sentiment, features, walk-forward models, backtest, charts,
  report generation.
- **Pinned dependencies**: `requirements.txt`.
- **Config template**: `.env.example` (no secrets; `.env` is gitignored).
- **Pre-computed tables**: `artifacts/` — six processed tables plus `scrape_summary.json`.
- **Report artefacts**: `reports/evaluation_report.pdf`, `.md`, `metrics_*.csv`,
  `sentiment_summary.md`, and the 8 interactive charts (HTML + PNG).
- **Tests**: 32 tests, including a leakage test that corrupts all data from bar `n` onwards
  and asserts the features at `n` are unchanged.

The raw scrape (~196 MB of Apify JSON) is **not** committed: it is too large for git and
redistributing raw X posts is avoidable. The committed `artifacts/` tables contain the
aggregated, model-ready data instead.

---

## Path A — rebuild the report from the committed tables

No API key, no Apify credit, no raw tweets.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/restore_artifacts.py
python run_pipeline.py --stage report
```

This recreates:

- `reports/metrics_prediction.csv`
- `reports/metrics_strategy.csv`
- `reports/figures/01…08` (HTML)
- `reports/sentiment_summary.md`
- `reports/evaluation_report.md`

The committed `reports/evaluation_report.pdf` is already the rendered version. To rebuild
the PDF and the PNG charts:

```bash
pip install playwright pypdf markdown
playwright install chromium
python scripts/export_report.py
```

`reports/cleaning_log.json` and the metrics CSVs are committed, so the report can be
assembled without re-running cleaning or training.

---

## Path B — refit the models from the committed hourly master table

No API key and no Apify credit, but the walk-forward models are trained again.

```bash
python scripts/restore_artifacts.py
python run_pipeline.py --stage train
python run_pipeline.py --stage report
```

`--stage train` rebuilds `features.parquet` from `master.parquet`, then refits Ridge,
LightGBM and the price-only ablations for h = 0…5 across the 4 walk-forward folds. Results
should match `reports/metrics_strategy.csv`; small differences are possible across
LightGBM/scikit-learn versions because the libraries are pinned only by lower bounds.

---

## Path C — full re-scrape from X

This is the only path that spends Apify credit.

```bash
cp .env.example .env          # Windows: copy .env.example .env
# fill APIFY_TOKEN and APIFY_ACTOR_ID in .env
python run_pipeline.py --stage ingest
python run_pipeline.py --stage process
python run_pipeline.py --stage train
python run_pipeline.py --stage report
```

Actor and query used:

- Actor: `wAusCMrm284Voaw86` (`xquik~x-tweet-scraper`)
- Query: `($BTC OR #Bitcoin OR bitcoin) lang:en -filter:replies -filter:retweets`
- Sort: `Top`, one window per day, 480 items/window, 60 days
- Cost: ~$0.15 per 1,000 tweets → ~$4.25 for ~28.3K tweets

**Credit note:** the Apify token in the submission email is a FREE-plan token with only a
small amount of credit left, so a full 60-day re-scrape will not run on that token. The
actor ID and query above are the handover; a fresh Apify token/account reproduces the
scrape. If you need the exact raw scrape without spending credit, the committed
`artifacts/` path above rebuilds every model and report number.

The scrape is idempotent: finished day-windows are skipped, and the pipeline stops safely
when the credit runs low.

---

## Expected results

| Metric | Value |
|---|---|
| Best forecast vs naive | RMSE ratio 1.001 (no model beats naive) |
| Best model strategy by Sharpe | `ridge_price_only__long_flat` |
| Net return / Sharpe / Sortino / Max DD | −16.5% / −7.93 / −10.16 / −17.7% |
| Buy-and-hold | −10.4% / −4.40 / −5.82 / −11.6% |
| `lgbm__long_short` cost illustration | +4.1% gross → −30.8% net, 40.8% fee drag |

The honest result is a null: no model beats the naive zero-return forecast, and every
strategy loses after 7 bps costs. The report states this rather than tuning it away.

---

## Tests

```bash
pytest -q
```

Expected: **32 passed**. The suite covers the leakage boundary, target construction,
walk-forward purge, metric formulas, fee/turnover accounting, the Apify adapter and the
yfinance fallback.
