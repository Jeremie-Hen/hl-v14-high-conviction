"""
v14 High-Conviction Meta-Strategy Bot — Configuration
24h horizon, 10x leverage, dynamic ATR exits, 4-signal conviction scoring.
"""
import os

# ── Hyperliquid credentials ──────────────────────────────────────
PRIVATE_KEY = os.getenv("HL_PRIVATE_KEY", "")
WALLET_ADDRESS = os.getenv("HL_WALLET_ADDRESS", "")

# ── Trading parameters ───────────────────────────────────────────
SYMBOL = os.getenv("SYMBOL", "BTC")
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
POSITION_EQUITY_PCT = float(os.getenv("POSITION_EQUITY_PCT", "1.0"))
MAX_POSITIONS = 1

# ── Strategy (v14 24h High Conviction) ───────────────────────────
CONVICTION_THRESHOLD = float(os.getenv("CONVICTION_THRESHOLD", "0.35"))
MAX_HOLD_HOURS = int(os.getenv("MAX_HOLD_HOURS", "24"))

# Signal weights (must sum to 1.0)
W_ML = 0.50
W_COUNTER = 0.20
W_TA = 0.15
W_CONSENSUS = 0.15

# Exit parameters
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT", "0.6"))
TRAIL_DISTANCE_PCT = float(os.getenv("TRAIL_DISTANCE_PCT", "0.3"))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "2.0"))
SL_MIN_PCT = 0.2
SL_MAX_PCT = 3.0

# Agent analysis
BAD_AGENT_THRESHOLD = 0.45
ACC_WINDOW = 72
MIN_BAD_AGENTS_FOR_SIGNAL = 2

# ML
MODEL_RETRAIN_HOUR = 0  # retrain daily at 00:00 UTC
MIN_TRAIN_SAMPLES = 200

# ── Fee structure (Hyperliquid base tier) ────────────────────────
MAKER_FEE_PCT = 0.00015
TAKER_FEE_PCT = 0.00045

# ── Bot settings ─────────────────────────────────────────────────
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"
TESTNET = os.getenv("TESTNET", "true").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# ── API endpoints ────────────────────────────────────────────────
MAINNET_API = "https://api.hyperliquid.xyz"
TESTNET_API = "https://api.hyperliquid-testnet.xyz"
API_URL = TESTNET_API if TESTNET else MAINNET_API

BINANCE_REST_URL = "https://api.binance.com/api/v3"
MYQUANT_API_URL = os.getenv(
    "MYQUANT_API_URL",
    "https://myquant.gg/api/agents/predictions?timeframe=1",
)

# ── Polling ──────────────────────────────────────────────────────
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "5"))
POLL_WINDOW_SEC = int(os.getenv("POLL_WINDOW_SEC", "600"))
POSITION_POLL_SEC = 60  # check position / trailing stop between hours
