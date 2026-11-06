# Pre-computed outputs

Small processed tables from the real run, committed so the evaluation report can be
reproduced without re-scraping X (the raw scrape is ~196 MB and stays out of git).

| file | rows | size |
|---|---:|---:|
| `master.parquet` | 1,464 | 151 KB |
| `features.parquet` | 1,440 | 735 KB |
| `predictions.parquet` | 4,032 | 203 KB |
| `strategy_returns.parquet` | 672 | 60 KB |
| `feature_importance.parquet` | 1,344 | 14 KB |
| `final_forecast.csv` | 6 | 1 KB |

Regenerate with `python run_pipeline.py --stage report` after a full run, or use
these directly: `master.parquet` -> `features.parquet` -> `predictions.parquet` /
`strategy_returns.parquet` are the exact inputs to the metrics and charts.
