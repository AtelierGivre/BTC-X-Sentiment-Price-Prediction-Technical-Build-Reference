from __future__ import annotations

import pandas as pd

from config import HORIZONS, PURGE
from src.models.walkforward import make_folds, make_ridge


def test_purge_covers_the_longest_target_horizon() -> None:
    assert PURGE >= max(HORIZONS)


def test_ridge_cv_also_purges_between_inner_folds() -> None:
    cv = make_ridge().named_steps["ridgecv"].cv
    assert cv.gap == PURGE


def test_gap_between_train_and_test() -> None:
    folds = list(make_folds(n=4000, train0=2000, test=168, step=168, gap=PURGE))
    assert len(folds) > 5
    for train, test in folds:
        assert test.start - train.stop >= PURGE
        assert train.start == 0
        assert test.stop <= 4000


def test_test_windows_move_forward_without_overlap() -> None:
    folds = list(make_folds(4000, 2000, 168, 168, PURGE))
    for (_, a), (_, b) in zip(folds, folds[1:]):
        assert b.start >= a.stop


def test_iloc_slices_have_exact_sizes() -> None:
    """Regression: .loc[slice] is end-inclusive and once leaked one extra row."""
    d = pd.DataFrame({"x": range(5000)})
    for train, test in make_folds(5000, 2000, 168, 168, PURGE):
        assert len(d.iloc[train]) == train.stop
        assert len(d.iloc[test]) == 168
        assert d.iloc[train].x.max() + PURGE < d.iloc[test].x.min() + 1
