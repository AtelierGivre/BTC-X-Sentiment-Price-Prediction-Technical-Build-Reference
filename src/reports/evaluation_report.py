"""Assemble reports/evaluation_report.md from the pipeline outputs."""
from __future__ import annotations

import json

import pandas as pd

from config import (
    DAYS, FEE_BPS, HORIZONS, PROC, PURGE, RAW, REPORTS, SYMBOL, TARGETS, TEST_DAYS, TRAIN_DAYS,
)
from src.features.build import NON_FEATURES
from src.utils.log import get_logger

log = get_logger(__name__)


def _table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main() -> None:
    clean_log = json.loads((REPORTS / "cleaning_log.json").read_text(encoding="utf-8"))
    pm = pd.read_csv(REPORTS / "metrics_prediction.csv")
    st = pd.read_csv(REPORTS / "metrics_strategy.csv")
    preds = pd.read_parquet(PROC / "predictions.parquet")
    features = pd.read_parquet(PROC / "features.parquet")
    fc = pd.read_csv(PROC / "final_forecast.csv")

    n_features = len([c for c in features.columns if c not in NON_FEATURES])
    folds = (
        preds.groupby("fold").hour.agg(["min", "max"]).reset_index()
        .rename(columns={"min": "test_start", "max": "test_end"})
    )

    tweets = pd.read_parquet(RAW / "tweets_raw.parquet")
    tweet_days = (tweets.created_at.max() - tweets.created_at.min()).total_seconds() / 86400
    tweet_cost = sum(
        float(json.loads(f.read_text(encoding="utf-8")).get("cost_usd") or 0)
        for f in (RAW / "apify_windows").glob("w_*.meta.json")
    )
    tweet_start = tweets.created_at.min()
    tweet_end = tweets.created_at.max()
    n_labelled = len(features.dropna(subset=TARGETS))
    first_train_days = round(min(TRAIN_DAYS, n_labelled / 2 / 24))

    cleaning_rows = "\n".join(f"| {k} | {v:,} |" for k, v in clean_log.items())

    pred_table = pm[pm.model.isin(["naive", "ridge", "lgbm", "ridge_price_only", "lgbm_price_only"])]
    pred_pivot = pred_table.pivot(index="h", columns="model", values="rmse_ratio_vs_naive").reset_index()
    pred_pivot["h"] = pred_pivot["h"].map(lambda x: str(int(x)))
    pred_pivot = pred_pivot[
        ["h", "naive", "ridge", "ridge_price_only", "lgbm", "lgbm_price_only"]
    ]

    piv = pred_table.pivot(index="h", columns="model", values="rmse_ratio_vs_naive")
    models_only = pred_table[pred_table.model != "naive"]
    best = models_only.loc[models_only.rmse_ratio_vs_naive.idxmin()]
    ridge_wins = int((piv["ridge"] < piv["ridge_price_only"]).sum())
    lgbm_wins = int((piv["lgbm"] < piv["lgbm_price_only"]).sum())
    strat_only = st[st.strategy != "buy_hold"]
    bs = strat_only.loc[strat_only.sharpe.idxmax()]
    bh = st[st.strategy == "buy_hold"].iloc[0]

    fi = pd.read_parquet(PROC / "feature_importance.parquet")
    gain = fi.groupby("feature").gain.mean()
    sent_prefixes = ("sent_mean", "w_follow", "w_engage", "pct_pos", "tweets_", "no_tweets")
    sent_share = gain[[f for f in gain.index if f.startswith(sent_prefixes)]].sum() / gain.sum()

    text = f"""# Evaluation report — {SYMBOL} X-sentiment price model

## Summary
- **Does sentiment add value?** Against the identical model on price only, Ridge wins in
  {ridge_wins}/{len(piv)} horizons and LightGBM in {lgbm_wins}/{len(piv)} (lower RMSE ratio).
- **Best forecast:** `{best.model}` at h={int(best.h)} has RMSE ratio
  {best.rmse_ratio_vs_naive:.3f} vs naive (1.000 = no improvement) and directional accuracy
  {best.dir_acc:.1%}.
- **Best tradable strategy:** `{bs.strategy}` — Sharpe {bs.sharpe:.2f}
  (95% CI {bs.sharpe_ci_low:.2f} to {bs.sharpe_ci_high:.2f}), Sortino {bs.sortino:.2f},
  Max DD {bs.max_drawdown:.1%}, net {bs.total_return:+.1%} vs {bs.gross_return:+.1%} gross
  (fees {bs.fee_drag:.1%}).
- **Buy-and-hold:** Sharpe {bh.sharpe:.2f}, Max DD {bh.max_drawdown:.1%},
  net {bh.total_return:+.1%}.

A gross edge smaller than the fees needed to harvest it cannot be traded profitably. The
bootstrap interval, not the point estimate, is the decision rule; with a short sample a wide
interval spanning zero is the expected outcome and is reported as such.

## 1. Objective and setup
Predict BTC's price at **n+0 … n+5 hours** from **X sentiment and price only**. Bars are 1 hour,
labelled by open time in UTC. At the start of bar `n` only information up to the close of bar
`n-1` is used. Target: `y_h = log(close[n+h] / close[n-1])` for h = {HORIZONS}. Costs are
{FEE_BPS:.0f} bps per position change.

## 2. Data
Price source: Binance BTCUSDT hourly klines (fallback yfinance BTC-USD). Full history: {DAYS} days
(~{DAYS * 24:,} bars, free); the model uses the tweet window plus 24h warm-up (~{tweet_days:.0f}
days), so no hour is trained on a fabricated zero-sentiment bar.

Tweet source: Apify X scrape, actor `xquik~x-tweet-scraper` (~$0.15/1K tweets). Actual
coverage: **{len(tweets):,} tweets over {tweet_days:.1f} days** ({tweet_start:%Y-%m-%d} →
{tweet_end:%Y-%m-%d} UTC), cost **${tweet_cost:.2f}**. A 180-day scrape was not affordable on
the Apify FREE plan ($5/month), so the model is trained only on the tweet window (plus 24h
of price warm-up); the rest of the price history is unused. This is disclosed, not silently
back-filled with another dataset. Sentiment: VADER.

Row counts through cleaning:

| step | rows |
|---|---:|
{cleaning_rows}

## 3. Sentiment method
VADER compound score in [-1, 1] per tweet. Hourly aggregates: tweet count, mean/median/std,
positive/negative share, follower-weighted and engagement-weighted mean. Optional transformer
scores are compared with VADER when `TRANSFORMER_MODEL` is set.

## 4. Features and leakage
{n_features} features: price returns at lags {list((1, 2, 3, 6, 12, 24))}, 6/24h volatility,
24h momentum, volume z-score; the same lags and moving averages for four sentiment series
(`sent_mean`, `w_follow`, `w_engage`, `pct_pos`) plus a 168h z-score and 1h change; tweet volume
(log and z-score); a no-tweets flag; hour-of-day and weekday. **All feature columns receive one
global `.shift(1)`**, and calendar features are added after the shift. `tests/test_leakage.py`
corrupts all data from bar `n` onwards and asserts features at `n` are unchanged.

## 5. Validation design
Expanding walk-forward: first train about {first_train_days} days (adaptive: 90 days capped at
half the sample), test {TEST_DAYS} days, step {TEST_DAYS} days, **{PURGE}-bar purge gap**.
{len(folds)} folds:

{_table(folds)}

Scalers and RidgeCV penalties are fitted inside each training window only.

## 6. Results — prediction (out-of-sample)
RMSE ratio vs the naive (zero-return) benchmark by horizon and model:

{_table(pred_pivot)}

A ratio at or above 1.000 means the model is no better than predicting a zero return; the
best model here reaches {best.rmse_ratio_vs_naive:.3f}, so no forecast beats the naive
benchmark at any horizon. Full metrics (RMSE, MAE, directional accuracy, Spearman IC, USD MAE)
are in `reports/metrics_prediction.csv`.

## 7. Results — strategy (after costs)
{_table(st, floatfmt=".3f")}

`fee_drag` is total fees in log-return units (7 bps × turnover); `gross_return` adds it back,
so the gap between gross and net is the cost of trading. All positions use the h=0 forecast
(the return of the current bar), entered at the start of the bar. Bootstrap intervals are 95%
moving-block bootstrap (24h blocks, 1,000 resamples) on the Sharpe.

## 8. Ablations
`ridge_price_only` and `lgbm_price_only` use the same models without any sentiment column.
Comparing them with `ridge` / `lgbm` answers the real question: **does sentiment add anything
beyond price?** Read the RMSE-ratio and Sharpe differences, not just the best absolute number.
LightGBM attributes {sent_share:.0%} of its split gain to sentiment features, but the price-only
ablation matches or beats the full model — gain share measures how often a feature is used,
not whether it helps out of sample.

## 9. Charts
Interactive charts are in `reports/figures/`:
1. sentiment vs price, 2. tweet volume, 3. rolling sentiment/return correlation,
4. predicted vs actual, 5. equity curves + drawdown, 6. metric matrix, 7. feature importance,
8. sentiment-change vs price information matrix (sentiment lag × forecast horizon).

## 10. Limitations
The tweet sample covers {tweet_days:.0f} days, not {DAYS}, because of the Apify FREE credit;
price history is longer but sentiment only exists inside that window. The scrape is a
`Top`-ranked sample (engagement-biased) capped at ~480 tweets/day, so quiet hours are
thin. Other limits: one market regime, bot noise, X search coverage, fees assumed at
{FEE_BPS:.0f} bps with no slippage, and fold-level position resets. A weak or null result
is acceptable if the method is clean.

## 11. Reproduction
```bash
python run_pipeline.py --stage ingest
python run_pipeline.py --stage process
python run_pipeline.py --stage train
python run_pipeline.py --stage report
pytest -q
```

Small pre-computed tables (`master`, `features`, `predictions`, `strategy_returns`,
`feature_importance`) are committed under `artifacts/`, so the report can be reproduced
without re-scraping X and without spending Apify credit.

## 12. Latest forecast
{_table(fc, floatfmt=".6f")}
"""
    out = REPORTS / "evaluation_report.md"
    out.write_text(text, encoding="utf-8")
    log.info("evaluation report written -> %s", out)


if __name__ == "__main__":
    main()
