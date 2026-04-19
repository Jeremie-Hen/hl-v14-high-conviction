"""
FeatureStore — append-only hourly snapshot of agent predictions + market state.

Every hourly cycle, the bot writes one row (one JSON line) with:
  - hour_ts             : ISO UTC hour boundary
  - btc_close           : BTC closing price at that hour
  - agent predictions   : per-agent {direction, confidence, predicted_price}
  - candle_features     : numeric TA / price features from ta_engine
  - ta_net_signal       : aggregated TA signal

This file is the ground truth used by retrain.py to build a daily refreshed
training set. Rows older than 24h can be labelled with `target_dir_24h` by
looking at BTC's close 24h forward (also a simple Binance lookup).

Storage: JSONL (line-delimited JSON) — append-only, crash-safe, easy to replay.
"""
import json
import os
from datetime import datetime, timezone

FEATURE_STORE_PATH = os.path.join(os.path.dirname(__file__), "feature_store.jsonl")


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def _sanitize(value):
    """JSON-safe scalars (avoid numpy / pandas types)."""
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    if isinstance(value, (int, float, bool, str)):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def append_hour(
    hour_ts: str,
    btc_close: float,
    agents: list[dict],
    candle_features: dict,
    ta_result: dict,
) -> bool:
    """Append a single hourly row to the feature store. Returns True on success."""
    try:
        row = {
            "hour_ts": hour_ts,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "btc_close": _sanitize(btc_close),
            "ta_net_signal": _sanitize(ta_result.get("ta_net_signal")),
            "ta_long": _sanitize(ta_result.get("ta_long")),
            "ta_short": _sanitize(ta_result.get("ta_short")),
            "candle_features": {k: _sanitize(v) for k, v in candle_features.items()},
            "agents": [
                {
                    "name": a["name"],
                    "direction": _sanitize(a.get("direction")),
                    "confidence": _sanitize(a.get("confidence")),
                    "predicted_price": _sanitize(a.get("predicted_price")),
                    "is_correct": _sanitize(a.get("is_correct")),
                }
                for a in (agents or [])
            ],
        }
        with open(FEATURE_STORE_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        return True
    except Exception as e:
        log(f"  [STORE] Error appending: {e}")
        return False


def read_all() -> list[dict]:
    """Read every row from the feature store. Returns [] if file missing."""
    if not os.path.exists(FEATURE_STORE_PATH):
        return []
    rows = []
    with open(FEATURE_STORE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def count_rows() -> int:
    if not os.path.exists(FEATURE_STORE_PATH):
        return 0
    try:
        with open(FEATURE_STORE_PATH, "r") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0
