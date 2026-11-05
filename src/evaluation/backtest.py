"""Turn out-of-sample predictions into a simulated strategy and score it.

At the start of bar n we look at the h=0 forecast (the return of the CURRENT bar):
  long_short : +1 if forecast > 0 else -1
  long_flat  : +1 if forecast > 0 else 0
Costs: FEE_BPS on every position change. Each test fold starts flat (conservative; positions
are not carried across the 6-bar purge gap between folds). Benchmark: buy-and-hold.

Output: reports/metrics_prediction.csv, reports/metrics_strategy.csv,
        data/processed/strategy_returns.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEE_BPS, PROC, REPORTS
from src.evaluation.metrics import (
    block_bootstrap_sharpe, max_drawdown, prediction_metrics, sharpe, sortino,
)
from src.utils.log import get_logger

log = get_logger(__name__)

MODELS = ["ridge", "lgbm", "ridge_price_only", "lgbm_price_only"]
ALL_MODELS = ["naive"] + MODELS


def strategy_returns(
    preds: pd.DataFrame, fee_bps: float = FEE_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    p = preds[preds.h == 0].sort_values("hour").reset_index(drop=True)
    out = pd.DataFrame({"hour": p.hour, "fold": p.fold, "buy_hold": p.y_true})
    trades: dict[str, int] = {}
    fees: dict[str, float] = {}

    for model in MODELS:
        for variant in ("long_short", "long_flat"):
            pos = np.sign(p[model]) if variant == "long_short" else (p[model] > 0).astype(float)
            change = pos.groupby(p.fold).diff().fillna(pos.abs())   # enter from flat each fold
            cost = fee_bps / 1e4 * change.abs()
            name = f"{model}__{variant}"
            out[name] = pos * p.y_true - cost
            trades[name] = int((change.abs() > 0).sum())
            fees[name] = float(cost.sum())

    trades["buy_hold"] = 1
    fees["buy_hold"] = 0.0
    return (
        out.set_index("hour"),
        pd.Series(trades, name="n_trades").to_frame(),
        pd.Series(fees, name="fee_drag"),
    )


def strategy_table(sr: pd.DataFrame, trades: pd.DataFrame, fees: pd.Series) -> pd.DataFrame:
    rows = []
    for col in [c for c in sr.columns if c != "fold"]:
        r = sr[col]
        lo, hi = block_bootstrap_sharpe(r)
        fee = float(fees.loc[col])
        rows.append({
            "strategy": col,
            "sharpe": sharpe(r),
            "sharpe_ci_low": lo,
            "sharpe_ci_high": hi,
            "sortino": sortino(r),
            "max_drawdown": max_drawdown(r),
            "total_return": float(np.exp(r.sum()) - 1),
            "gross_return": float(np.exp(r.sum() + fee) - 1),
            "fee_drag": fee,
            "n_trades": int(trades.loc[col, "n_trades"]),
            "n_hours": len(r),
        })
    return pd.DataFrame(rows)


def main() -> None:
    preds = pd.read_parquet(PROC / "predictions.parquet")
    pm = prediction_metrics(preds, ALL_MODELS)
    pm.to_csv(REPORTS / "metrics_prediction.csv", index=False)

    sr, trades, fees = strategy_returns(preds)
    st = strategy_table(sr, trades, fees)
    st.to_csv(REPORTS / "metrics_strategy.csv", index=False)
    sr.to_parquet(PROC / "strategy_returns.parquet")

    log.info("strategy metrics:\n%s", st.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
