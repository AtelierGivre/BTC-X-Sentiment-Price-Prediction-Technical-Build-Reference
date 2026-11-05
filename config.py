"""Central configuration. All paths and knobs live here.

Conventions (keep these true everywhere):
  * 1-hour bars, labelled by OPEN time, UTC in code and storage.
  * Decision time for row n is the START of bar n.
  * Features for row n may use only information up to the CLOSE of bar n-1.
  * Target: y_h = log(close[n+h] / close[n-1]) for h = 0..5.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DEMO = os.getenv("DEMO", "0") == "1"

# --- paths -----------------------------------------------------------------
DATA = ROOT / ("data_demo" if DEMO else "data")
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROC = DATA / "processed"
REPORTS = ROOT / ("reports_demo" if DEMO else "reports")
FIGS = REPORTS / "figures"
LOGS = ROOT / "logs"
for _p in (RAW, INTERIM, PROC, REPORTS, FIGS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# --- secrets ---------------------------------------------------------------
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "").strip()
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TRANSFORMER_MODEL = os.getenv("TRANSFORMER_MODEL", "").strip()

# --- data ------------------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
YF_SYMBOL = os.getenv("YF_SYMBOL", "BTC-USD")
DAYS = int(os.getenv("DAYS", "180"))                 # price history
TWEET_DAYS = int(os.getenv("TWEET_DAYS", "60"))      # tweet coverage (budget-limited)
SEARCH_QUERY = os.getenv(
    "SEARCH_QUERY",
    "($BTC OR #Bitcoin OR bitcoin) lang:en -filter:replies -filter:retweets",
)
WINDOWS_PER_DAY = int(os.getenv("WINDOWS_PER_DAY", "1"))
MAX_ITEMS_PER_WINDOW = int(os.getenv("MAX_ITEMS_PER_WINDOW", "480"))
WINDOW_HOURS = max(1, 24 // WINDOWS_PER_DAY)
APIFY_BUDGET_USD = float(os.getenv("APIFY_BUDGET_USD", "4.50"))

# --- sentiment -------------------------------------------------------------
SENT_COL = "s_tx" if TRANSFORMER_MODEL else "s_vader"
TX_MAX_PER_HOUR = int(os.getenv("TX_MAX_PER_HOUR", "20"))
POS_THRESH = 0.05

# --- modelling -------------------------------------------------------------
HORIZONS = [0, 1, 2, 3, 4, 5]
TARGETS = [f"y{h}" for h in HORIZONS]
TRAIN_DAYS = 90
TEST_DAYS = 7
STEP_DAYS = 7
PURGE = 6
SEED = 42

# --- evaluation ------------------------------------------------------------
FEE_BPS = 7.0
PERIODS_PER_YEAR = 24 * 365
