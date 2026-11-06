# Evaluation report — BTCUSDT X-sentiment price model

## Summary
- **Does sentiment add value?** Against the identical model on price only, Ridge wins in
  0/6 horizons and LightGBM in 2/6 (lower RMSE ratio).
- **Best forecast:** `ridge_price_only` at h=0 has RMSE ratio
  1.001 vs naive (1.000 = no improvement) and directional accuracy
  48.2%.
- **Best model strategy by Sharpe (all are loss-making):** `ridge_price_only__long_flat` — Sharpe -7.93
  (95% CI -13.28 to -2.26), Sortino -10.16,
  Max DD -17.7%, net -16.5% vs -10.7% gross
  (fees 6.7%).
- **Buy-and-hold:** Sharpe -4.40, Max DD -11.6%,
  net -10.4%.

A gross edge smaller than the fees needed to harvest it cannot be traded profitably. The
bootstrap interval, not the point estimate, is the decision rule; with a short sample a wide
interval spanning zero is the expected outcome and is reported as such.

## 1. Objective and setup
Predict BTC's price at **n+0 … n+5 hours** from **X sentiment and price only**. Bars are 1 hour,
labelled by open time in UTC. At the start of bar `n` only information up to the close of bar
`n-1` is used. Target: `y_h = log(close[n+h] / close[n-1])` for h = [0, 1, 2, 3, 4, 5]. Costs are
7 bps per position change.

## 2. Data
Price source: Binance BTCUSDT hourly klines (fallback yfinance BTC-USD). Full history: 180 days
(~4,320 bars, free); the model uses the tweet window plus 24h warm-up (~60
days), so no hour is trained on a fabricated zero-sentiment bar.

Tweet source: Apify X scrape, actor `xquik~x-tweet-scraper` (~$0.15/1K tweets). Actual
coverage: **28,342 tweets over 60.0 days** (2026-09-06 →
2026-11-05 UTC), cost **$4.25**. A 180-day scrape was not affordable on
the Apify FREE plan ($5/month), so the model is trained only on the tweet window (plus 24h
of price warm-up); the rest of the price history is unused. This is disclosed, not silently
back-filled with another dataset. Sentiment: VADER.

Row counts through cleaning:

| step | rows |
|---|---:|
| start | 28,342 |
| after_dedup_tweet_id | 28,342 |
| after_nonempty | 28,331 |
| after_english_filter | 28,312 |
| after_author_repeats | 28,291 |
| after_swarm_dedup | 28,291 |
| after_spam_filter | 28,291 |

## 3. Sentiment method
VADER compound score in [-1, 1] per tweet. Hourly aggregates: tweet count, mean/median/std,
positive/negative share, follower-weighted and engagement-weighted mean. Optional transformer
scores are compared with VADER when `TRANSFORMER_MODEL` is set.

## 4. Features and leakage
56 features: price returns at lags [1, 2, 3, 6, 12, 24], 6/24h volatility,
24h momentum, volume z-score; the same lags and moving averages for four sentiment series
(`sent_mean`, `w_follow`, `w_engage`, `pct_pos`) plus a 168h z-score and 1h change; tweet volume
(log and z-score); a no-tweets flag; hour-of-day and weekday. **All feature columns receive one
global `.shift(1)`**, and calendar features are added after the shift. `tests/test_leakage.py`
corrupts all data from bar `n` onwards and asserts features at `n` are unchanged.

## 5. Validation design
Expanding walk-forward: first train about 30 days (adaptive: 90 days capped at
half the sample), test 7 days, step 7 days, **6-bar purge gap**.
4 folds:

|   fold | test_start          | test_end            |
|-------:|:--------------------|:--------------------|
|      0 | 2026-10-06 10:00:00 | 2026-10-13 09:00:00 |
|      1 | 2026-10-13 10:00:00 | 2026-10-20 09:00:00 |
|      2 | 2026-10-20 10:00:00 | 2026-10-27 09:00:00 |
|      3 | 2026-10-27 10:00:00 | 2026-11-03 09:00:00 |

Scalers and RidgeCV penalties are fitted inside each training window only.

## 6. Results — prediction (out-of-sample)
RMSE ratio vs the naive (zero-return) benchmark by horizon and model:

|   h |   naive |   ridge |   ridge_price_only |   lgbm |   lgbm_price_only |
|----:|--------:|--------:|-------------------:|-------:|------------------:|
|   0 |   1.000 |   1.001 |              1.001 |  1.028 |             1.031 |
|   1 |   1.000 |   1.002 |              1.002 |  1.053 |             1.047 |
|   2 |   1.000 |   1.003 |              1.003 |  1.063 |             1.064 |
|   3 |   1.000 |   1.004 |              1.004 |  1.083 |             1.072 |
|   4 |   1.000 |   1.005 |              1.004 |  1.092 |             1.071 |
|   5 |   1.000 |   1.006 |              1.005 |  1.090 |             1.071 |

A ratio at or above 1.000 means the model is no better than predicting a zero return; the
best model here reaches 1.001, so no forecast beats the naive
benchmark at any horizon. Full metrics (RMSE, MAE, directional accuracy, Spearman IC, USD MAE)
are in `reports/metrics_prediction.csv`.

## 7. Results — strategy (after costs)
`sharpe` is the **Sharpe Ratio**, `sortino` is the **Sortino Ratio**, and `max_drawdown` is the
**Max Drawdown** (peak-to-trough on the compounded equity curve). All are annualised from
hourly net returns.

| strategy                     |   sharpe |   sharpe_ci_low |   sharpe_ci_high |   sortino |   max_drawdown |   total_return |   gross_return |   fee_drag |   n_trades |   n_hours |
|:-----------------------------|---------:|----------------:|-----------------:|----------:|---------------:|---------------:|---------------:|-----------:|-----------:|----------:|
| buy_hold                     |   -4.401 |          -9.298 |            1.900 |    -5.816 |         -0.116 |         -0.104 |         -0.104 |      0.000 |          1 |       672 |
| ridge__long_short            |  -21.301 |         -29.206 |          -14.159 |   -24.999 |         -0.424 |         -0.416 |         -0.085 |      0.449 |        321 |       672 |
| ridge__long_flat             |  -15.149 |         -21.567 |           -8.694 |   -18.647 |         -0.286 |         -0.277 |         -0.094 |      0.225 |        321 |       672 |
| lgbm__long_short             |  -14.546 |         -19.158 |           -9.096 |   -18.598 |         -0.313 |         -0.308 |          0.041 |      0.408 |        292 |       672 |
| lgbm__long_flat              |  -12.653 |         -17.841 |           -7.302 |   -16.193 |         -0.221 |         -0.212 |         -0.034 |      0.204 |        291 |       672 |
| ridge_price_only__long_short |  -10.019 |         -15.514 |           -5.001 |   -12.590 |         -0.237 |         -0.222 |         -0.111 |      0.134 |         96 |       672 |
| ridge_price_only__long_flat  |   -7.928 |         -13.281 |           -2.259 |   -10.155 |         -0.177 |         -0.165 |         -0.107 |      0.067 |         96 |       672 |
| lgbm_price_only__long_short  |  -16.690 |         -22.391 |          -10.471 |   -20.907 |         -0.354 |         -0.348 |         -0.026 |      0.401 |        287 |       672 |
| lgbm_price_only__long_flat   |  -14.135 |         -18.822 |           -8.818 |   -17.727 |         -0.239 |         -0.235 |         -0.066 |      0.200 |        286 |       672 |

`fee_drag` is total fees in log-return units (7 bps × turnover); `gross_return` adds it back,
so the gap between gross and net is the cost of trading. All positions use the h=0 forecast
(the return of the current bar), entered at the start of the bar. Bootstrap intervals are 95%
moving-block bootstrap (24h blocks, 1,000 resamples) on the Sharpe.

## 8. Ablations
`ridge_price_only` and `lgbm_price_only` use the same models without any sentiment column.
Comparing them with `ridge` / `lgbm` answers the real question: **does sentiment add anything
beyond price?** Read the RMSE-ratio and Sharpe differences, not just the best absolute number.
LightGBM attributes 55% of its total split gain to sentiment features, but the
price-only ablation matches or beats the full model — gain share measures how much a feature
reduces training loss, not whether it helps out of sample.

## 9. Charts
Interactive charts are in `reports/figures/`:
1. sentiment vs price, 2. tweet volume, 3. rolling sentiment/return correlation,
4. predicted vs actual, 5. equity curves + drawdown, 6. metric matrix, 7. feature importance,
8. sentiment-change vs price information matrix (sentiment lag × forecast horizon).

## 10. Limitations
The tweet sample covers 60 days, not 180, because of the Apify FREE credit;
price history is longer but sentiment only exists inside that window. The scrape is a
`Top`-ranked sample (engagement-biased) capped at ~480 tweets/day, so quiet hours are
thin. Other limits: one market regime, bot noise, X search coverage, fees assumed at
7 bps with no slippage, and positions carried across fold boundaries (test
windows are contiguous; the final open position is marked to market with no exit fee). A
weak or null result is acceptable if the method is clean.

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

The forecast below is fitted on all labelled rows and predicts the next unlabelled bar. It is
therefore **in-sample** and shown for illustration only; it is not part of the out-of-sample
evaluation above.

| bar                       |   h |   ridge_ret |   ridge_price |   lgbm_ret |   lgbm_price |
|:--------------------------|----:|------------:|--------------:|-----------:|-------------:|
| 2026-11-05 06:00:00+00:00 |   0 |    0.000094 |  80407.539294 |   0.000736 | 80459.209331 |
| 2026-11-05 06:00:00+00:00 |   1 |   -0.000074 |  80394.046471 |   0.001395 | 80512.269142 |
| 2026-11-05 06:00:00+00:00 |   2 |   -0.000127 |  80389.801139 |   0.000969 | 80477.980326 |
| 2026-11-05 06:00:00+00:00 |   3 |   -0.000009 |  80399.301677 |   0.001203 | 80496.797381 |
| 2026-11-05 06:00:00+00:00 |   4 |    0.000181 |  80414.561472 |   0.003505 | 80682.321035 |
| 2026-11-05 06:00:00+00:00 |   5 |    0.000383 |  80430.801789 |   0.003069 | 80647.153394 |
