"""Eight interactive Plotly charts (HTML; PNG only if kaleido is installed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import FIGS, HORIZONS, PROC, REPORTS
from src.utils.log import get_logger

log = get_logger(__name__)
EST = "America/New_York"


def _save(fig: go.Figure, name: str) -> None:
    fig.write_html(FIGS / f"{name}.html", include_plotlyjs="directory")
    try:                                            # optional PNG export
        fig.write_image(FIGS / f"{name}.png", scale=2)
    except Exception:
        pass
    log.info("chart saved: %s", name)


def chart_sentiment_vs_price() -> None:
    m = pd.read_parquet(PROC / "master.parquet").set_index("hour")
    m.index = m.index.tz_convert(EST)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=m.index, y=m.close, name="BTC close (USD)", line=dict(color="#1f77b4")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=m.index, y=m.w_follow, name="follower-weighted sentiment",
                             line=dict(color="#ff7f0e", width=1)), secondary_y=True)
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment (-1..1)", secondary_y=True)
    fig.update_layout(title="BTC price vs X sentiment", hovermode="x unified", height=520)
    _save(fig, "01_sentiment_vs_price")


def chart_tweet_volume() -> None:
    m = pd.read_parquet(PROC / "master.parquet").set_index("hour")
    m.index = m.index.tz_convert(EST)
    fig = go.Figure(go.Bar(x=m.index, y=m.n_tweets, name="tweets / hour", marker_color="#9467bd"))
    fig.update_layout(title="Tweet volume per hour", xaxis_title="time (EST)", yaxis_title="tweets")
    _save(fig, "02_tweet_volume")


def chart_rolling_correlation() -> None:
    m = pd.read_parquet(PROC / "master.parquet").set_index("hour")
    ret_next = np.log(m.close).diff().shift(-1)
    corr = m.sent_mean.rolling(168, min_periods=48).corr(ret_next)
    idx = corr.index.tz_convert(EST)
    fig = go.Figure(go.Scatter(x=idx, y=corr, name="rolling corr (168h)"))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(title="Sentiment vs next-hour return (168h rolling Pearson corr)",
                      yaxis_title="correlation", height=420)
    _save(fig, "03_rolling_correlation")


def chart_pred_vs_actual() -> None:
    preds = pd.read_parquet(PROC / "predictions.parquet")
    p0 = preds[preds.h == 0]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("OOS scatter (h=0)", "Last 7 days (h=0)"))
    for model, color in (("ridge", "#1f77b4"), ("lgbm", "#d62728")):
        fig.add_trace(go.Scatter(x=p0.y_true, y=p0[model], mode="markers", name=model,
                                 marker=dict(size=4, opacity=0.5, color=color)), row=1, col=1)
    lim = float(np.nanpercentile(np.abs(p0[["y_true", "ridge", "lgbm"]].values), 99))
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="perfect",
                             line=dict(dash="dash", color="grey")), row=1, col=1)

    last = p0.sort_values("hour").tail(24 * 7)
    idx = pd.to_datetime(last.hour, utc=True).dt.tz_convert(EST)
    fig.add_trace(go.Scatter(x=idx, y=last.y_true, name="actual"), row=1, col=2)
    fig.add_trace(go.Scatter(x=idx, y=last.ridge, name="ridge"), row=1, col=2)
    fig.add_trace(go.Scatter(x=idx, y=last.lgbm, name="lgbm"), row=1, col=2)
    fig.update_layout(title="Predicted vs actual log-return (out-of-sample)", height=520)
    _save(fig, "04_pred_vs_actual")


def chart_equity() -> None:
    sr = pd.read_parquet(PROC / "strategy_returns.parquet")
    strategy_cols = [c for c in sr.columns if c != "fold"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=("Equity curves", "Drawdown (best strategy)"))
    for col in strategy_cols:
        eq = np.exp(sr[col].cumsum())
        fig.add_trace(go.Scatter(x=sr.index, y=eq, name=col), row=1, col=1)
    best = max(strategy_cols, key=lambda c: float(np.exp(sr[c].sum())))
    eq = np.exp(sr[best].cumsum())
    dd = eq / eq.cummax() - 1
    fig.add_trace(go.Scatter(x=sr.index, y=dd, name=f"drawdown ({best})", fill="tozeroy"),
                  row=2, col=1)
    fig.update_layout(title="Strategy backtest (7 bps per position change)", height=640)
    _save(fig, "05_equity_curves")


def chart_metric_matrix() -> None:
    pm = pd.read_csv(REPORTS / "metrics_prediction.csv")
    pm = pm[pm.model == "ridge"]
    matrix = pm.pivot(index="h", columns="model", values="rmse_ratio_vs_naive")
    acc = pm.pivot(index="h", columns="model", values="dir_acc")
    ic = pm.pivot(index="h", columns="model", values="ic")
    frame = pd.DataFrame({
        "RMSE ratio vs naive": matrix.iloc[:, 0],
        "Directional accuracy": acc.iloc[:, 0],
        "Spearman IC": ic.iloc[:, 0],
    })
    fig = go.Figure(go.Heatmap(
        z=frame.values, x=frame.columns, y=[f"n+{h}" for h in frame.index],
        text=np.round(frame.values, 3), texttemplate="%{text}", colorscale="RdYlGn", zmid=0,
    ))
    fig.update_layout(title="Metric matrix — Ridge (out-of-sample)", height=420)
    _save(fig, "06_metric_matrix")


def chart_feature_importance() -> None:
    path = PROC / "feature_importance.parquet"
    if not path.exists():
        return
    imp = pd.read_parquet(path)
    top = (imp.groupby("feature", as_index=False).gain.mean()
           .sort_values("gain", ascending=False).head(20))
    fig = go.Figure(go.Bar(x=top.gain, y=top.feature, orientation="h", marker_color="#2ca02c"))
    fig.update_layout(title="LightGBM feature importance (mean gain across folds/horizons)",
                      yaxis=dict(autorange="reversed"), height=560)
    _save(fig, "07_feature_importance")


def chart_sentiment_change_matrix() -> None:
    """The brief's "visual matrix": how sentiment relates to price across lags and horizons."""
    m = pd.read_parquet(PROC / "master.parquet").set_index("hour")
    base = np.log(m.close.shift(1))
    lags = [0, 1, 2, 3, 6, 12, 24]
    z = []
    for lag in lags:
        s = m.sent_mean.shift(lag)
        z.append([s.corr(np.log(m.close.shift(-h)) - base, method="spearman") for h in HORIZONS])
    fig = go.Figure(go.Heatmap(
        z=np.array(z), x=[f"n+{h}" for h in HORIZONS], y=[f"sent lag {l}h" for l in lags],
        text=np.round(np.array(z), 3), texttemplate="%{text}",
        colorscale="RdBu_r", zmid=0, zmin=-0.12, zmax=0.12,
        colorbar_title="Spearman IC",
    ))
    fig.update_layout(
        title="Sentiment change vs price: information matrix (out-of-sample signal is ~0)",
        xaxis_title="forecast horizon", yaxis_title="sentiment lag", height=460,
    )
    _save(fig, "08_sentiment_change_matrix")


def main() -> None:
    chart_sentiment_vs_price()
    chart_tweet_volume()
    chart_rolling_correlation()
    chart_pred_vs_actual()
    chart_equity()
    chart_metric_matrix()
    chart_feature_importance()
    chart_sentiment_change_matrix()


if __name__ == "__main__":
    main()
