"""All metric formulas in one place, so they are easy to read and unit-test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PERIODS_PER_YEAR, SEED

ANN = np.sqrt(PERIODS_PER_YEAR)


def _flat(sd: float) -> bool:
    """True when a standard deviation is numerically zero.

    Floating-point summation on a constant series can return ~1e-17 instead of 0,
    which would blow the ratio up to ~1e17. Compare against the series scale.
    """
    return (not np.isfinite(sd)) or np.isclose(sd, 0.0, atol=1e-12)


def sharpe(r) -> float:
    r = pd.Series(r).dropna()
    sd = float(r.std())
    return float("nan") if len(r) < 2 or _flat(sd) else float(r.mean() / sd * ANN)


def sortino(r) -> float:
    """Downside deviation = sqrt(mean(min(r, 0)^2)), target 0 (textbook definition)."""
    r = pd.Series(r).dropna()
    dd = float(np.sqrt((np.minimum(r, 0) ** 2).mean()))
    return float("nan") if _flat(dd) else float(r.mean() / dd * ANN)


def max_drawdown(r) -> float:
    """Worst peak-to-trough fall of the equity curve, starting from equity 1.0.

    Log returns -> exp(cumsum). The explicit 1.0 start matters: without it a loss in
    the very first bar is never compared with the initial capital and is missed.
    """
    eq = np.concatenate([[1.0], np.exp(np.cumsum(pd.Series(r).fillna(0).values))])
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def block_bootstrap_sharpe(r, block: int = 24, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """95% CI of the Sharpe via moving-block bootstrap (keeps short-term autocorrelation)."""
    values = np.asarray(pd.Series(r).dropna())
    n = len(values)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([values[i:i + block] for i in starts])[:n]
        sd = sample.std()
        sharpes.append(sample.mean() / sd * ANN if sd > 0 else np.nan)
    lo, hi = np.nanpercentile(sharpes, [2.5, 97.5])
    return float(lo), float(hi)


def prediction_metrics(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Per model x horizon, out-of-sample only."""
    rows = []
    for h, g in preds.groupby("h"):
        naive_rmse = np.sqrt(((g.y_true - g["naive"]) ** 2).mean())
        for model in models:
            err = g.y_true - g[model]
            rmse = np.sqrt((err ** 2).mean())
            nonzero = g.y_true != 0
            ic = spearmanr(g[model], g.y_true)[0] if g[model].std() > 0 else np.nan
            price_mae = (
                g.close_prev * np.exp(g[model]) - g.close_prev * np.exp(g.y_true)
            ).abs().mean()
            dir_acc = (
                float((np.sign(g[model][nonzero]) == np.sign(g.y_true[nonzero])).mean())
                if model != "naive" else np.nan
            )
            rows.append({
                "model": model, "h": h, "rmse": rmse, "mae": err.abs().mean(),
                "rmse_ratio_vs_naive": rmse / naive_rmse,
                "dir_acc": dir_acc, "ic": ic, "price_mae_usd": price_mae,
            })
    return pd.DataFrame(rows)
