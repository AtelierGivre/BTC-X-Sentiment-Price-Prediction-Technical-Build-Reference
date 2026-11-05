"""Walk-forward (expanding window) training and out-of-sample prediction.

   |------ train (expanding) ------| gap=PURGE |-- test (7 days) --|
   |---------- train (bigger) ------------| gap |-- test (next 7 days) --|

* one model per horizon (direct strategy): 6 horizons x 4 models
* models: naive (predict 0), Ridge, LightGBM, plus the same two WITHOUT sentiment
* scalers and hyperparameters are fitted on the TRAIN part of each fold only

Output: data/processed/predictions.parquet, feature_importance.parquet, final_forecast.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import HORIZONS, PROC, PURGE, SEED, STEP_DAYS, TARGETS, TEST_DAYS, TRAIN_DAYS
from src.features.build import feature_columns, split_features
from src.utils.log import get_logger

log = get_logger(__name__)


def make_folds(n: int, train0: int, test: int, step: int, gap: int):
    """Yield (train_slice, test_slice). Train always ends `gap` rows before test starts."""
    s = train0
    while s + gap + test <= n:
        yield slice(0, s), slice(s + gap, s + gap + test)
        s += step


def make_ridge():
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-1, 4, 12), cv=TimeSeriesSplit(3)),
    )


def make_lgbm():
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        random_state=SEED, n_jobs=2, verbose=-1,
    )


def model_specs(df: pd.DataFrame) -> dict:
    feats = feature_columns(df)
    price_cols, _ = split_features(feats)
    return {
        "ridge": (make_ridge, feats),
        "lgbm": (make_lgbm, feats),
        "ridge_price_only": (make_ridge, price_cols),
        "lgbm_price_only": (make_lgbm, price_cols),
    }


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.dropna(subset=TARGETS).reset_index(drop=True)      # drops the forecast row
    hours = d.hour.diff().dropna()
    assert (hours == pd.Timedelta("1h")).all(), "feature rows are not contiguous hours"

    specs = model_specs(d)
    n = len(d)
    train0 = min(TRAIN_DAYS * 24, int(n * 0.5))
    folds = list(make_folds(n, train0, TEST_DAYS * 24, STEP_DAYS * 24, PURGE))
    log.info("walk-forward: %d rows | first train=%d | %d folds | gap=%d", n, train0, len(folds), PURGE)

    parts: list[pd.DataFrame] = []
    importance: list[dict] = []

    for k, (tr, te) in enumerate(folds):
        assert tr.stop + PURGE <= te.start                     # never let train reach the test
        d_tr, d_te = d.iloc[tr], d.iloc[te]                    # .iloc = positional (end-exclusive)
        for h in HORIZONS:
            out = pd.DataFrame({
                "hour": d_te.hour.values,
                "fold": k,
                "h": h,
                "y_true": d_te[f"y{h}"].values,
                "close_prev": d_te.close_prev.values,
                "naive": 0.0,
            })
            for name, (maker, cols) in specs.items():
                model = maker().fit(d_tr[cols], d_tr[f"y{h}"])
                out[name] = model.predict(d_te[cols])
                if name == "lgbm":
                    for feat, gain in zip(cols, model.feature_importances_):
                        importance.append({"h": h, "fold": k, "feature": feat, "gain": float(gain)})
            parts.append(out)
        log.info(
            "fold %d/%d | train ends %s | test %s -> %s",
            k + 1, len(folds), d_tr.hour.iloc[-1], d_te.hour.iloc[0], d_te.hour.iloc[-1],
        )

    return pd.concat(parts, ignore_index=True), pd.DataFrame(importance)


def final_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Fit on ALL labelled rows and predict the single forecast row ('now')."""
    train = df.dropna(subset=TARGETS)
    now = df[df[TARGETS].isna().all(axis=1)].tail(1)
    specs = model_specs(df)
    rows = []
    for h in HORIZONS:
        row = {"bar": now.hour.iloc[0], "h": h}
        for name in ("ridge", "lgbm"):
            maker, cols = specs[name]
            pred = maker().fit(train[cols], train[f"y{h}"]).predict(now[cols])[0]
            row[f"{name}_ret"] = float(pred)
            row[f"{name}_price"] = float(now.close_prev.iloc[0] * np.exp(pred))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(PROC / "features.parquet")
    preds, importance = run(df)
    preds.to_parquet(PROC / "predictions.parquet", index=False)
    importance.to_parquet(PROC / "feature_importance.parquet", index=False)
    forecast = final_forecast(df)
    forecast.to_csv(PROC / "final_forecast.csv", index=False)
    log.info("predictions saved: %d rows | forecast bar: %s", len(preds), forecast.bar.iloc[0])


if __name__ == "__main__":
    main()
