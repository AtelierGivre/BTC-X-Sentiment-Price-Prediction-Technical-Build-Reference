"""Restore the committed pre-computed tables into `data/processed/`.

    python scripts/restore_artifacts.py
    python run_pipeline.py --stage report

This is the no-credit reproduction path: it needs no `.env`, no Apify token and no raw
tweets. Run `--stage train` first if you also want to refit the models from the committed
`master.parquet`.

Overwrites the matching files in `data/processed/` with the committed snapshot.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

FILES = [
    "master.parquet",
    "features.parquet",
    "predictions.parquet",
    "strategy_returns.parquet",
    "feature_importance.parquet",
    "final_forecast.csv",
]


def main() -> None:
    missing = [name for name in FILES if not (ARTIFACTS / name).exists()]
    if missing:
        raise SystemExit(
            "missing artifacts: " + ", ".join(missing)
            + "\nRun the pipeline once with real data first."
        )

    PROC.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(ARTIFACTS / name, PROC / name)
        print(f"restored  data/processed/{name}")

    news = ARTIFACTS / "news.json"
    if news.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        shutil.copy2(news, RAW / "news.json")
        print("restored  data/raw/news.json")

    print("\nNext: python run_pipeline.py --stage report")
    print("  or: python run_pipeline.py --stage train && python run_pipeline.py --stage report")


if __name__ == "__main__":
    main()
