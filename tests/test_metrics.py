from __future__ import annotations

import numpy as np

from src.evaluation.metrics import ANN, max_drawdown, sharpe, sortino


def test_sharpe_is_nan_for_constant_returns() -> None:
    assert np.isnan(sharpe([0.01] * 100))


def test_sharpe_sign() -> None:
    r = np.array([0.01] * 50 + [-0.005] * 50)
    assert sharpe(r) > 0


def test_sortino_uses_the_downside_deviation_formula() -> None:
    r = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert np.isclose(sortino(r), r.mean() / downside * ANN)


def test_sortino_differs_from_sharpe_when_upside_is_large() -> None:
    # same mean/left tail, but large upside moves inflate the Sharpe denominator
    r = np.array([0.10, -0.01, 0.10, -0.01])
    assert sortino(r) > sharpe(r)


def test_max_drawdown() -> None:
    # equity: 1 -> exp(0.1) -> exp(0.1-0.2); trough is exp(-0.2) below the peak
    dd = max_drawdown([0.1, -0.2])
    assert np.isclose(dd, np.exp(-0.2) - 1), dd


def test_max_drawdown_counts_a_loss_in_the_first_bar() -> None:
    # the starting equity of 1.0 is the peak; a first-bar loss is a real drawdown
    dd = max_drawdown([-0.1, 0.05])
    assert np.isclose(dd, np.exp(-0.1) - 1), dd


def test_max_drawdown_is_zero_when_monotone() -> None:
    assert np.isclose(max_drawdown([0.01, 0.02, 0.03]), 0.0)
