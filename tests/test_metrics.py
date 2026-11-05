from __future__ import annotations

import numpy as np

from src.evaluation.metrics import max_drawdown, sharpe, sortino


def test_sharpe_is_nan_for_constant_returns() -> None:
    assert np.isnan(sharpe([0.01] * 100))


def test_sharpe_sign() -> None:
    r = np.array([0.01] * 50 + [-0.005] * 50)
    assert sharpe(r) > 0


def test_sortino_ignores_upside_volatility() -> None:
    r = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
    assert sortino(r) > sharpe(r) * 0, "sortino should be finite and positive"


def test_max_drawdown() -> None:
    # equity: 1 -> exp(0.1) -> exp(0.1-0.2); trough is exp(-0.2) below the peak
    dd = max_drawdown([0.1, -0.2])
    assert np.isclose(dd, np.exp(-0.2) - 1), dd


def test_max_drawdown_is_zero_when_monotone() -> None:
    assert np.isclose(max_drawdown([0.01, 0.02, 0.03]), 0.0)
