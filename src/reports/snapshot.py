"""Copy the small processed tables into `artifacts/` for offline reproduction.

The raw scrape is ~196 MB and the Apify credit is spent, so the full `data/` tree stays
gitignored. These five tables (~1 MB) are enough to rebuild every number in the report
without touching Apify.
"""
from __future__ import annotations

import shutil

import pandas as pd

from config import PROC, ROOT
from src.utils.log import get_logger

log = get_logger(__name__)

FILES = [
    "master.parquet",
    "features.parquet",
    "predictions.parquet",
    "strategy_returns.parquet",
    "feature_importance.parquet",
    "final_forecast.csv",
]


def main() -> None:
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    rows = []
    for name in FILES:
        src = PROC / name
        if not src.exists():
            log.warning("snapshot | missing %s", name)
            continue
        shutil.copy2(src, out / name)
        if src.suffix == ".parquet":
            n = len(pd.read_parquet(src))
        else:
            n = max(0, sum(1 for _ in src.open(encoding="utf-8")) - 1)
        rows.append(f"| `{name}` | {n:,} | {(out / name).stat().st_size / 1024:.0f} KB |")

    text = (
        "# Pre-computed outputs\n\n"
        "Small processed tables from the real run, committed so the evaluation report can be\n"
        "reproduced without re-scraping X (the raw scrape is ~196 MB and stays out of git).\n\n"
        "| file | rows | size |\n|---|---:|---:|\n"
        + "\n".join(rows)
        + "\n\nRegenerate with `python run_pipeline.py --stage report` after a full run, or use\n"
        "these directly: `master.parquet` -> `features.parquet` -> `predictions.parquet` /\n"
        "`strategy_returns.parquet` are the exact inputs to the metrics and charts.\n"
    )
    (out / "README.md").write_text(text, encoding="utf-8")
    log.info("snapshot: %d tables -> %s", len(rows), out)


if __name__ == "__main__":
    main()
