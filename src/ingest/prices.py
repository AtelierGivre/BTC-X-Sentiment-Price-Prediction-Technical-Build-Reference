"""Hourly BTC price ingestion.

Primary source: Binance public klines (no key, max 1000 rows per call -> paginate).
Fallback: yfinance (BTC-USD, 1h). Binance is tried first; if it is blocked or rate-limited
the fallback is automatic.

Output: data/raw/prices_1h.parquet with columns hour[UTC], open, high, low, close, volume.
Bars are labelled by OPEN time. The current, still-incomplete bar is excluded.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from config import DAYS, RAW, SYMBOL, YF_SYMBOL
from src.utils.log import get_logger

log = get_logger(__name__)

BINANCE_ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
]
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "tb_base", "tb_quote", "ignore",
]
NUM_COLUMNS = ["open", "high", "low", "close", "volume"]


def _fetch_binance(days: int, symbol: str) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("h")          # start of the current, incomplete bar
    start = end - pd.Timedelta(days=days)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list] = []
    last_error: Exception | None = None

    for url in BINANCE_ENDPOINTS:
        rows = []
        cur = int(start.timestamp() * 1000)
        try:
            while cur < end_ms:
                r = requests.get(
                    url,
                    params={
                        "symbol": symbol,
                        "interval": "1h",
                        "startTime": cur,
                        "endTime": end_ms - 1,      # exclude the current, incomplete bar
                        "limit": 1000,
                    },
                    timeout=30,
                )
                if r.status_code in (418, 429, 451):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                rows += batch
                cur = batch[-1][0] + 3_600_000
                time.sleep(0.15)
            if rows:
                break
        except Exception as exc:                          # try the next endpoint
            last_error = exc
            log.warning("Binance endpoint failed (%s): %s", url, exc)

    if not rows:
        raise RuntimeError(f"all Binance endpoints failed: {last_error}")

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df["hour"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["hour"] + NUM_COLUMNS].astype({c: float for c in NUM_COLUMNS})
    return df.drop_duplicates("hour").sort_values("hour").reset_index(drop=True)


def _fetch_yfinance(days: int) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        YF_SYMBOL, period=f"{days}d", interval="1h",
        auto_adjust=False, progress=False, threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no hourly data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower).reset_index()
    time_col = "datetime" if "datetime" in df.columns else df.columns[0]
    df["hour"] = pd.to_datetime(df[time_col], utc=True).dt.floor("h")
    df = df.rename(columns={"adj close": "close"})
    df = df[["hour"] + NUM_COLUMNS].astype({c: float for c in NUM_COLUMNS})
    return df.drop_duplicates("hour").sort_values("hour").reset_index(drop=True)


def fetch_prices(days: int = DAYS, symbol: str = SYMBOL, source: str = "auto") -> pd.DataFrame:
    """Return an hourly OHLCV frame. source: auto | binance | yfinance."""
    if source in ("auto", "binance"):
        try:
            return _fetch_binance(days, symbol)
        except Exception as exc:
            if source == "binance":
                raise
            log.warning("Binance unavailable (%s); using yfinance", exc)
    return _fetch_yfinance(days)


def check_prices(df: pd.DataFrame, days: int | None = None) -> None:
    """Fail loudly if there are gaps, duplicates or NaNs. A silent gap corrupts every lag."""
    assert not df.empty, "price frame is empty"
    assert df.hour.is_monotonic_increasing and df.hour.is_unique, "hours must be sorted and unique"
    gaps = df.hour.diff().dropna()
    bad = gaps[gaps != pd.Timedelta("1h")]
    assert bad.empty, f"{len(bad)} gaps in the hourly index, first: {bad.head(3).to_dict()}"
    assert df[NUM_COLUMNS].notna().all().all(), "NaN in OHLCV"
    assert (df[NUM_COLUMNS] > 0).all().all(), "non-positive price/volume"
    if days is not None:
        assert len(df) >= days * 24 - 2, f"only {len(df)} bars, expected about {days * 24}"


def main() -> None:
    df = fetch_prices()
    check_prices(df, DAYS)
    out = RAW / "prices_1h.parquet"
    df.to_parquet(out, index=False)
    log.info("prices saved: %d rows | %s -> %s | gaps=0", len(df), df.hour.min(), df.hour.max())
    print(f"prices -> {out}")


if __name__ == "__main__":
    main()
