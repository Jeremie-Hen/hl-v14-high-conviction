"""
Signal Engine — v14 High-Conviction Meta-Strategy
Combines 4 signal sources into a conviction score with dynamic exit parameters.

Components (v14 24h High Conviction weights):
  1. LightGBM P(UP) probability  — w=0.50
  2. Counter-trade bad agents     — w=0.20
  3. TA confluence                — w=0.15
  4. Quality-weighted consensus   — w=0.15

Ported from s11_meta_strategy.py (backtest version) for live inference.
"""
import json
import os
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_v14.pkl")
STATE_PATH = os.path.join(os.path.dirname(__file__), "signal_state.json")

FORBIDDEN_FEATURES = {
    "target_dir", "target_ret", "actual_return_pct", "agent_closing_price",
    "target_ret_1h", "target_dir_1h", "target_ret_4h", "target_dir_4h",
    "target_ret_8h", "target_dir_8h", "target_ret_24h", "target_dir_24h",
    "regime",
}


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


# ═════════════════════════════════════════════════════════════════════
# ML MODEL
# ═════════════════════════════════════════════════════════════════════

class MLModel:
    """LightGBM model manager with expanding-window retraining."""

    def __init__(self):
        self.model = None
        self.feature_cols: list[str] = []
        self.medians: dict[str, float] = {}
        self.last_train_ts: str | None = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.feature_cols = data["feature_cols"]
                self.medians = data["medians"]
                self.last_train_ts = data.get("last_train_ts")
                log(f"  [ML] Loaded model ({len(self.feature_cols)} features, "
                    f"trained {self.last_train_ts})")
            except Exception as e:
                log(f"  [ML] Error loading model: {e}")

    def save(self):
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "feature_cols": self.feature_cols,
                    "medians": self.medians,
                    "last_train_ts": self.last_train_ts,
                }, f)
        except Exception as e:
            log(f"  [ML] Error saving model: {e}")

    def train(self, history_df: pd.DataFrame) -> bool:
        """
        Train on historical data. Expects a DataFrame with feature columns
        and 'target_dir_24h' as the target (or 'target_dir' fallback).
        """
        if not HAS_LGB:
            log("  [ML] LightGBM not available")
            return False

        target_col = "target_dir_24h" if "target_dir_24h" in history_df.columns else "target_dir"
        if target_col not in history_df.columns:
            log("  [ML] No target column found")
            return False

        feature_cols = [
            c for c in history_df.columns
            if c not in FORBIDDEN_FEATURES
            and not c.endswith("_correct")
            and history_df[c].dtype in ("float64", "float32", "int64", "int32")
        ]

        X = history_df[feature_cols].copy()
        nan_pct = X.isna().mean()
        keep = nan_pct[nan_pct < 0.5].index.tolist()
        X = X[keep]

        medians = X.median()
        X = X.fillna(medians)

        var = X.var()
        keep = var[var > 1e-10].index.tolist()
        X = X[keep]

        if len(keep) < 5 or len(X) < config.MIN_TRAIN_SAMPLES:
            log(f"  [ML] Insufficient data: {len(X)} rows, {len(keep)} features")
            return False

        y = (history_df.loc[X.index, target_col] == 1).astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            verbose=-1, random_state=42, importance_type="gain",
        )
        model.fit(X, y)

        train_proba = model.predict_proba(X)[:, 1]
        train_acc = ((train_proba > 0.5).astype(int) == y.values).mean()

        self.model = model
        self.feature_cols = keep
        self.medians = medians.to_dict()
        self.last_train_ts = datetime.now(timezone.utc).isoformat()
        self.save()

        log(f"  [ML] Trained: {len(X)} samples, {len(keep)} features, "
            f"train acc={train_acc:.1%}")
        return True

    def predict_proba_up(self, features: dict) -> float | None:
        """Return P(UP) for a single observation. Returns None if model not ready."""
        if self.model is None or not self.feature_cols:
            return None

        row = {}
        for col in self.feature_cols:
            val = features.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row[col] = self.medians.get(col, 0.0)
            else:
                row[col] = val

        X = pd.DataFrame([row], columns=self.feature_cols)
        proba = self.model.predict_proba(X)[:, 1]
        return float(proba[0])


# ═════════════════════════════════════════════════════════════════════
# COUNTER-TRADE SIGNAL
# ═════════════════════════════════════════════════════════════════════

def compute_counter_trade(agent_features: dict) -> float:
    """
    Invert bad-agent consensus.
    Returns +1 (counter LONG), -1 (SHORT), 0 (no signal).
    """
    bad_votes = []
    for key, val in agent_features.items():
        if not key.endswith("_dir"):
            continue
        agent_id = key.replace("_dir", "")
        acc_key = f"{agent_id}_acc_{config.ACC_WINDOW}h"
        acc = agent_features.get(acc_key)
        if acc is None:
            continue
        if acc <= config.BAD_AGENT_THRESHOLD:
            bad_votes.append(-val)

    if len(bad_votes) < config.MIN_BAD_AGENTS_FOR_SIGNAL:
        return 0.0

    vote_sum = sum(bad_votes)
    return float(np.sign(vote_sum))


# ═════════════════════════════════════════════════════════════════════
# WEIGHTED CONSENSUS
# ═════════════════════════════════════════════════════════════════════

def compute_weighted_consensus(agent_features: dict) -> float:
    """
    Quality-weighted vote: weight = (accuracy - 0.5).
    Returns normalized signal in [-1, 1].
    """
    weighted_sum = 0.0
    total_abs_w = 0.0

    for key, direction in agent_features.items():
        if not key.endswith("_dir"):
            continue
        if direction == 0:
            continue
        agent_id = key.replace("_dir", "")
        acc_key = f"{agent_id}_acc_{config.ACC_WINDOW}h"
        acc = agent_features.get(acc_key)
        if acc is None:
            continue

        w = acc - 0.5
        weighted_sum += direction * w
        total_abs_w += abs(w)

    if total_abs_w == 0:
        return 0.0

    return max(-1.0, min(1.0, weighted_sum / total_abs_w))


# ═════════════════════════════════════════════════════════════════════
# HOUR FILTER
# ═════════════════════════════════════════════════════════════════════

class HourFilter:
    """Tracks which hours have negative expected returns in live data."""

    def __init__(self):
        self.bad_hours: set[int] = set()
        self._load()

    def _load(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r") as f:
                    data = json.load(f)
                self.bad_hours = set(data.get("bad_hours", []))
                if self.bad_hours:
                    log(f"  [HOUR] Bad hours loaded: {sorted(self.bad_hours)}")
            except Exception:
                pass

    def save(self):
        try:
            with open(STATE_PATH, "w") as f:
                json.dump({"bad_hours": sorted(self.bad_hours)}, f)
        except Exception:
            pass

    def update_from_history(self, history_df: pd.DataFrame):
        """Learn bad hours from accumulated trade history DataFrame."""
        target_col = "target_dir_24h" if "target_dir_24h" in history_df.columns else "target_dir"
        if "hour_of_day" not in history_df.columns or target_col not in history_df.columns:
            return
        if "consensus_dir" not in history_df.columns:
            return

        bad = set()
        for h in range(24):
            mask = (history_df["hour_of_day"] == h) & (history_df["consensus_dir"] != 0)
            if mask.sum() < 20:
                continue
            sub = history_df.loc[mask]
            acc = (sub["consensus_dir"] == sub[target_col]).mean()
            if acc < 0.47:
                bad.add(h)

        self.bad_hours = bad
        self.save()
        if bad:
            log(f"  [HOUR] Updated bad hours: {sorted(bad)}")

    def is_allowed(self, hour: int) -> bool:
        return hour not in self.bad_hours


# ═════════════════════════════════════════════════════════════════════
# CONVICTION SCORER
# ═════════════════════════════════════════════════════════════════════

def compute_signal(
    ml_model: MLModel,
    hour_filter: HourFilter,
    candle_features: dict,
    agent_features: dict,
    ta_result: dict,
    current_hour: int,
    atr_pct: float = 0.5,
) -> dict | None:
    """
    Combine all 4 signal sources into a conviction score.
    Returns signal dict or None if no trade.
    """
    # 1. ML signal
    ml_proba = ml_model.predict_proba_up(candle_features | agent_features)
    ml_sig = 0.0
    if ml_proba is not None:
        ml_sig = (ml_proba - 0.5) * 2  # map [0,1] -> [-1,1]

    # 2. Counter-trade signal
    counter_sig = compute_counter_trade(agent_features)

    # 3. TA confluence
    ta_sig = ta_result.get("ta_net_signal", 0.0)

    # 4. Weighted consensus
    consensus_sig = compute_weighted_consensus(agent_features)

    # Weighted conviction
    conviction = (
        config.W_ML * ml_sig
        + config.W_COUNTER * counter_sig
        + config.W_TA * ta_sig
        + config.W_CONSENSUS * consensus_sig
    )

    strength = abs(conviction)
    direction = 1 if conviction > 0 else -1

    # Always log the component breakdown so we can verify every signal
    # source is contributing, even on hours where no trade is taken.
    log(f"  [SIG] Conviction={conviction:+.3f} "
        f"(ML={ml_sig:+.3f} CT={counter_sig:+.1f} TA={ta_sig:+.3f} CS={consensus_sig:+.3f})")

    # Hour filter
    if not hour_filter.is_allowed(current_hour):
        log(f"  [SIG] Hour {current_hour} filtered out (bad hour)")
        return None

    # Conviction threshold
    if strength < config.CONVICTION_THRESHOLD:
        log(f"  [SIG] Below threshold: |{conviction:.3f}| < {config.CONVICTION_THRESHOLD}")
        return None

    # Dynamic exit parameters
    sl_pct = max(config.SL_MIN_PCT, min(config.SL_MAX_PCT, atr_pct * config.SL_MULTIPLIER))

    log(f"  [SIG] SIGNAL: {'LONG' if direction == 1 else 'SHORT'} "
        f"conviction={conviction:+.3f}")

    return {
        "direction": "LONG" if direction == 1 else "SHORT",
        "direction_int": direction,
        "conviction": conviction,
        "strength": strength,
        "components": {
            "ml": ml_sig,
            "ml_proba": ml_proba,
            "counter": counter_sig,
            "ta": ta_sig,
            "consensus": consensus_sig,
        },
        "sl_pct": sl_pct,
        "trail_activate_pct": config.TRAIL_ACTIVATE_PCT,
        "trail_distance_pct": config.TRAIL_DISTANCE_PCT,
        "max_hold_hours": config.MAX_HOLD_HOURS,
        "leverage": config.LEVERAGE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
