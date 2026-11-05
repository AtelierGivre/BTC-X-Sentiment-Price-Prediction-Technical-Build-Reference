from __future__ import annotations

import numpy as np

from tests.test_leakage import fake_master
from src.features.build import build


def test_target_definition() -> None:
    m = fake_master()
    d = build(m)
    row = d.iloc[50]
    i = m.index[m.hour == row.hour][0]
    assert np.isclose(row.y0, np.log(m.close[i] / m.close[i - 1]))
    assert np.isclose(row.y5, np.log(m.close[i + 5] / m.close[i - 1]))


def test_close_prev_is_previous_bar() -> None:
    m = fake_master()
    d = build(m)
    row = d.iloc[100]
    i = m.index[m.hour == row.hour][0]
    assert np.isclose(row.close_prev, m.close[i - 1])
