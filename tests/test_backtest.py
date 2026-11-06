"""Backtest accounting: fees are charged on turnover and reported separately."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import strategy_returns, strategy_table


def _preds() -> pd.DataFrame:
    y = [0.01, 0.01, -0.01, 0.01, -0.01, -0.01]
    ridge = [0.01, -0.01, 0.01, 0.01, -0.01, 0.01]
    return pd.DataFrame({
        "hour": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
        "fold": 0,
        "h": 0,
        "y_true": y,
        "naive": 0.0,
        "ridge": ridge,
        "lgbm": ridge,
        "ridge_price_only": ridge,
        "lgbm_price_only": ridge,
    })


def test_fees_charged_on_turnover() -> None:
    sr, trades, fees = strategy_returns(_preds(), fee_bps=7)

    # positions +1,-1,+1,+1,-1,+1 -> 9 units of turnover including entry from flat
    assert trades.loc["ridge__long_short", "n_trades"] == 5      # bars with a change
    assert fees["ridge__long_short"] == pytest.approx(9 * 7e-4)
    assert fees["buy_hold"] == 0.0


def test_positions_are_carried_across_fold_boundaries() -> None:
    """A position held through a contiguous fold boundary must not pay a second entry fee."""
    p = _preds()
    p["fold"] = [0, 0, 0, 1, 1, 1]
    for col in ["ridge", "lgbm", "ridge_price_only", "lgbm_price_only"]:
        p[col] = 0.01                       # always long

    _, trades, fees = strategy_returns(p, fee_bps=7)

    assert trades.loc["ridge__long_short", "n_trades"] == 1
    assert fees["ridge__long_short"] == pytest.approx(7e-4)
    assert fees["ridge__long_flat"] == pytest.approx(7e-4)


def test_gross_equals_net_plus_fees() -> None:
    sr, trades, fees = strategy_returns(_preds(), fee_bps=7)
    st = strategy_table(sr, trades, fees).set_index("strategy")

    for name in ["ridge__long_short", "ridge__long_flat", "lgbm__long_short"]:
        net = float(np.exp(sr[name].sum()) - 1)
        assert st.loc[name, "total_return"] == pytest.approx(net)
        assert st.loc[name, "gross_return"] == pytest.approx(
            float(np.exp(np.log1p(net) + fees[name]) - 1)
        )
        assert st.loc[name, "fee_drag"] == pytest.approx(fees[name])
