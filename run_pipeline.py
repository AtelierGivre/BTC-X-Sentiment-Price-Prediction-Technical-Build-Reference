"""Run the pipeline stage by stage.

    python run_pipeline.py --stage ingest     # prices + Apify tweets
    python run_pipeline.py --stage process    # clean -> sentiment -> hourly master
    python run_pipeline.py --stage train      # features -> walk-forward models
    python run_pipeline.py --stage report     # metrics, charts, summary, report
    python run_pipeline.py --stage all

Add --demo to run on synthetic data with no API keys. Demo numbers are NOT real results.
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", choices=["ingest", "process", "train", "report", "all"])
ap.add_argument("--demo", action="store_true", help="use synthetic data and separate folders")
args = ap.parse_args()
if args.demo:
    os.environ["DEMO"] = "1"          # must be set BEFORE config is imported


def ingest() -> None:
    if args.demo:
        from src.ingest import synthetic
        synthetic.main()
    else:
        from src.ingest import prices, tweets_apify
        prices.main()
        tweets_apify.main([])


def process() -> None:
    from src.process import aggregate, clean, sentiment
    clean.main()
    sentiment.main()
    aggregate.main()


def train() -> None:
    from src.features import build
    from src.models import walkforward
    build.main()
    walkforward.main()


def report() -> None:
    from src.evaluation import backtest
    from src.reports import evaluation_report, snapshot, summary
    from src.visualization import charts
    backtest.main()
    charts.main()
    summary.main()
    evaluation_report.main()
    snapshot.main()


STAGES = {"ingest": ingest, "process": process, "train": train, "report": report}

if __name__ == "__main__":
    for name in (STAGES if args.stage == "all" else [args.stage]):
        print(f"\n===== STAGE: {name} =====")
        STAGES[name]()
