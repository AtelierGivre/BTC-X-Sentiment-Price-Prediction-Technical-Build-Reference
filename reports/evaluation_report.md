# Evaluation report — BTCUSDT X-sentiment price model

## Summary
- **Does sentiment add value?** Against the identical model on price only, Ridge wins in
  0/6 horizons and LightGBM in 2/6 (lower RMSE ratio).
- **Best forecast:** `ridge_price_only` at h=0 has RMSE ratio
  1.001 vs naive (1.000 = no improvement) and directional accuracy
  48.2%.
- **Best tradable strategy:** `ridge_price_only__long_flat` — Sharpe -8.02
  (95% CI -13.37 to -2.32), Sortino -10.27,
  Max DD -17.9%, net -16.7% vs -10.7% gross
  (fees 6.9%).
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
| ridge__long_short            |  -21.331 |         -29.255 |          -14.068 |   -25.026 |         -0.424 |         -0.416 |         -0.085 |      0.449 |        323 |       672 |
| ridge__long_flat             |  -15.180 |         -21.615 |           -8.653 |   -18.676 |         -0.287 |         -0.277 |         -0.094 |      0.225 |        322 |       672 |
| lgbm__long_short             |  -14.569 |         -19.176 |           -9.135 |   -18.621 |         -0.314 |         -0.309 |          0.041 |      0.409 |        294 |       672 |
| lgbm__long_flat              |  -12.646 |         -17.807 |           -7.362 |   -16.179 |         -0.221 |         -0.212 |         -0.034 |      0.204 |        291 |       672 |
| ridge_price_only__long_short |  -10.102 |         -15.608 |           -5.079 |   -12.689 |         -0.238 |         -0.224 |         -0.111 |      0.136 |         99 |       672 |
| ridge_price_only__long_flat  |   -8.020 |         -13.369 |           -2.323 |   -10.267 |         -0.179 |         -0.167 |         -0.107 |      0.069 |         99 |       672 |
| lgbm_price_only__long_short  |  -16.657 |         -22.295 |          -10.464 |   -20.872 |         -0.354 |         -0.347 |         -0.026 |      0.400 |        288 |       672 |
| lgbm_price_only__long_flat   |  -14.136 |         -18.808 |           -8.826 |   -17.728 |         -0.239 |         -0.235 |         -0.066 |      0.200 |        286 |       672 |

`fee_drag` is total fees in log-return units (7 bps × turnover); `gross_return` adds it back,
so the gap between gross and net is the cost of trading. All positions use the h=0 forecast
(the return of the current bar), entered at the start of the bar. Bootstrap intervals are 95%
moving-block bootstrap (24h blocks, 1,000 resamples) on the Sharpe.

## 8. Ablations
`ridge_price_only` and `lgbm_price_only` use the same models without any sentiment column.
Comparing them with `ridge` / `lgbm` answers the real question: **does sentiment add anything
beyond price?** Read the RMSE-ratio and Sharpe differences, not just the best absolute number.
LightGBM attributes 58% of its split gain to sentiment features, but the price-only
ablation matches or beats the full model — gain share measures how often a feature is used,
not whether it helps out of sample.

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
7 bps with no slippage, and fold-level position resets. A weak or null result
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
| bar                       |   h |   ridge_ret |   ridge_price |   lgbm_ret |   lgbm_price |
|:--------------------------|----:|------------:|--------------:|-----------:|-------------:|
| 2026-11-05 06:00:00+00:00 |   0 |    0.000094 |  80407.539294 |   0.000736 | 80459.209331 |
| 2026-11-05 06:00:00+00:00 |   1 |   -0.000074 |  80394.046471 |   0.001395 | 80512.269142 |
| 2026-11-05 06:00:00+00:00 |   2 |   -0.000127 |  80389.801139 |   0.000969 | 80477.980326 |
| 2026-11-05 06:00:00+00:00 |   3 |   -0.000009 |  80399.301677 |   0.001203 | 80496.797381 |
| 2026-11-05 06:00:00+00:00 |   4 |    0.000181 |  80414.561472 |   0.003505 | 80682.321035 |
| 2026-11-05 06:00:00+00:00 |   5 |    0.000383 |  80430.801789 |   0.003069 | 80647.153394 |
