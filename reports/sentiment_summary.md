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
- **Current price of Bitcoin for Nov. 4, 2026 | Fortune** — https://fortune.com/article/price-of-bitcoin-11-04-2026  
  Current price of Bitcoin for Nov. 4, 2026 Getty Images Add Fortune on Google for similar content. At 7 a.m. Eastern Time today, one Bitcoin (BTC) is priced at $78,520.27. That’s up
- **Bitcoin Price Forecast: BTC/USD 12.5% Plunge Nears Critical Support** — https://www.stonex.com/en-gb/news-and-analysis/bitcoin-price-forecast-btc-usd-12-5pct-plunge-nears-critical-support  
  A closer look at the Bitcoin daily chart shows BTC/USD trading within the confines of a proposed descending pitchfork with price testing the lower parallel today. This slope conver
- **J.P. Morgan Bitcoin Price Prediction: No Target Cited** — https://bitcoinfoundation.org/news/analysis/market-coverage-cites-no-j-p-morgan-target-as-btc-trades-near-77k  
  CoinGecko’s November 3 listing puts Bitcoin at $77,435, with a market capitalization of about $1.56 trillion and 24-hour volume of approximately $27.8 billion. The figures refer to
- **Bitcoin Price Prediction for November 2026** — https://finance.yahoo.com/markets/crypto/articles/bitcoin-price-prediction-november-2026-204712855.html  
  Currently priced at $78,050, Bitcoin has seen a slight 0.7% uptick in the last 24 hours. It peaked at $86,489 on October 4 before hitting a daily low of $76,141 on October 30, when
- **Federal Reserve Opts to Leave Rates Unchanged** — https://investingnews.com/cryptocurrency-market-recap  
  Bitcoin (BTC) was priced at US$78,004.26, trading flat over the past 24 hours. ### Bitcoin price chart Image 8 _Bitcoin price performance, November 2, 2026._ _Chart via the Investi

## Outlook
Sentiment over the last day is rising at +0.223, with volume at 20 tweets/hour.
The model's near-term path is shown above; the spread between Ridge and LightGBM is a useful
indicator of disagreement. Any edge here is small and regime-dependent, and the backtest should
be read alongside its bootstrap interval and transaction costs.

*Caveats: short sample, bot noise, scraper sampling limits, and fees with no slippage model.*
