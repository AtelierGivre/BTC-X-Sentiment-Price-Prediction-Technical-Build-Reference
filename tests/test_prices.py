from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingest.prices import check_prices


def frame(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "hour": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "open": np.ones(n), "high": np.ones(n) * 1.1, "low": np.ones(n) * 0.9,
        "close": np.ones(n), "volume": np.ones(n) * 10,
    })


def test_valid_frame_passes() -> None:
    check_prices(frame())


def test_gap_fails_loudly() -> None:
    df = frame()
    df = df.drop(index=5).reset_index(drop=True)
    with pytest.raises(AssertionError):
        check_prices(df)


def test_duplicate_fails_loudly() -> None:
    df = pd.concat([frame(), frame().iloc[[5]]], ignore_index=True)
    df = df.sort_values("hour").reset_index(drop=True)
    with pytest.raises(AssertionError):
        check_prices(df)
