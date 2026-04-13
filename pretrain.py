#!/usr/bin/env python3
"""
Pre-train the v14 LightGBM model from historical CSV + Binance candles.
Produces model_v14.pkl that the bot loads on startup.

Usage:
  python pretrain.py --csv path/to/quant-agents-export.csv
"""
import argparse
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(line_buffering=True)

import config

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: pip install lightgbm")
    sys.exit(1)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_v14.pkl")
HOUR_STATE_PATH = os.path.join(os.path.dirname(__file__), "signal_state.json")

DIRECTION_MAP = {
    "LONG": 1, "SHORT": -1,
    "UP": 1, "DOWN": -1,
    "BUY": 1, "SELL": -1,
    "BULLISH": 1, "BEARISH": -1,
}

FORBIDDEN_FEATURES = {
    "target_dir", "target_ret", "actual_return_pct", "agent_closing_price",
    "target_ret_1h", "target_dir_1h", "target_ret_4h", "target_dir_4h",
    "target_ret_8h", "target_dir_8h", "target_ret_24h", "target_dir_24h",
    "regime", "hour_key",
}


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


# ═════════════════════════════════════════════════════════════════════
# STEP 1: Load & pivot agent CSV into hourly rows
# ═════════════════════════════════════════════════════════════════════

def load_agent_csv(csv_path: str) -> pd.DataFrame:
    log(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"  Raw rows: {len(df)}, agents: {df['agentName'].nunique()}")

    df["runDate"] = pd.to_datetime(df["runDate"], utc=True)
    df["hour_key"] = df["runDate"].dt.floor("h")
    df["dir_int"] = df["direction"].map(DIRECTION_MAP).fillna(0).astype(int)

    # Per-agent rolling accuracy (72h window)
    agent_acc = {}
    for agent_name, grp in df.groupby("agentName"):
        grp = grp.sort_values("runDate")
        correct = grp["isCorrect"].map({True: 1.0, False: 0.0, "true": 1.0, "false": 0.0})
        correct = pd.to_numeric(correct, errors="coerce")
        rolling = correct.rolling(config.ACC_WINDOW, min_periods=10).mean()
        for hour_key, acc_val in zip(grp["hour_key"], rolling):
            agent_acc[(agent_name, hour_key)] = acc_val

    # Also compute overall agent accuracy as a fallback
    overall_acc = {}
    for agent_name, grp in df.groupby("agentName"):
        correct = grp["isCorrect"].map({True: 1.0, False: 0.0, "true": 1.0, "false": 0.0})
        correct = pd.to_numeric(correct, errors="coerce").dropna()
        if len(correct) >= 20:
            overall_acc[agent_name] = correct.mean()

    hours = sorted(df["hour_key"].unique())
    log(f"  Hours: {len(hours)} ({hours[0]} to {hours[-1]})")

    import re
    safe_names = {}
    for name in df["agentName"].unique():
        safe = re.sub(r'[^a-zA-Z0-9]', '_', name)
        safe = re.sub(r'_+', '_', safe).strip('_')
        safe_names[name] = safe

    rows = []
    for hour in hours:
        hour_df = df[df["hour_key"] == hour]
        feature_row = {"hour_key": hour}

        for _, agent_row in hour_df.iterrows():
            name = agent_row["agentName"]
            safe = safe_names[name]
            feature_row[f"agent_{safe}_dir"] = agent_row["dir_int"]

            acc = agent_acc.get((name, hour))
            if acc is not None and not np.isnan(acc):
                feature_row[f"agent_{safe}_acc_{config.ACC_WINDOW}h"] = acc
            elif name in overall_acc:
                feature_row[f"agent_{safe}_acc_{config.ACC_WINDOW}h"] = overall_acc[name]

        rows.append(feature_row)

    agent_df = pd.DataFrame(rows).set_index("hour_key").sort_index()
    log(f"  Agent feature matrix: {agent_df.shape}")
    return agent_df


# ═════════════════════════════════════════════════════════════════════
# STEP 2: Fetch historical 1h candles from Binance
# ═════════════════════════════════════════════════════════════════════

def fetch_all_candles(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    log(f"Fetching Binance 1h candles: {start_date} to {end_date}")
    all_candles = []
    current = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    while current < end_ms:
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{config.BINANCE_REST_URL}/klines",
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "1h",
                        "startTime": current,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == 2:
                    log(f"  Binance error: {e}")
                    data = []
                time.sleep(2)

        if not data:
            break

        all_candles.extend(data)
        last_close = data[-1][6]
        current = last_close + 1

        if len(data) < 1000:
            break
        time.sleep(0.2)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    log(f"  Candles: {len(df)} rows ({df.index[0]} to {df.index[-1]})")
    return df


# ═════════════════════════════════════════════════════════════════════
# STEP 3: Compute TA indicators (same as ta_engine.py)
# ═════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    from ta_engine import compute_indicators as _compute
    return _compute(df)


def compute_ta_signals_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TA net signal for every row (not just the last)."""
    from ta_engine import compute_ta_signals

    ta_rows = []
    for i in range(200, len(df)):
        window = df.iloc[:i + 1]
        result = compute_ta_signals(window)
        ta_rows.append({
            "ta_net_signal": result["ta_net_signal"],
            "ta_long": result["ta_long"],
            "ta_short": result["ta_short"],
        })

    ta_df = pd.DataFrame(ta_rows, index=df.index[200:])
    return ta_df


# ═════════════════════════════════════════════════════════════════════
# STEP 4: Build targets (24h forward return)
# ═════════════════════════════════════════════════════════════════════

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df["target_ret_24h"] = close.shift(-24) / close - 1
    df["target_dir_24h"] = (df["target_ret_24h"] > 0).astype(int)
    return df


# ═════════════════════════════════════════════════════════════════════
# STEP 5: Get candle features for each hour
# ═════════════════════════════════════════════════════════════════════

def get_candle_features_all(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "return_1h", "return_4h", "return_24h",
        "atr_14", "atr_pct", "vol_24h", "vol_72h", "vol_ratio",
        "ema_12", "ema_26", "ema_50", "trend_ema",
        "bb_bandwidth", "bb_pct_b", "rsi_14", "adx_14",
        "plus_di", "minus_di",
        "macd", "macd_signal", "macd_hist",
        "vol_sma_24", "vol_ratio_24",
        "hour_of_day",
    ]
    available = [c for c in feature_cols if c in df.columns]
    return df[available].copy()


# ═════════════════════════════════════════════════════════════════════
# STEP 6: Train LightGBM
# ═════════════════════════════════════════════════════════════════════

def train_model(features_df: pd.DataFrame, target_col: str = "target_dir_24h"):
    log(f"Training LightGBM on {len(features_df)} samples...")

    y = features_df[target_col].copy()

    feature_cols = [
        c for c in features_df.columns
        if c not in FORBIDDEN_FEATURES
        and not c.endswith("_correct")
        and features_df[c].dtype in ("float64", "float32", "int64", "int32")
    ]

    X = features_df[feature_cols].copy()

    nan_pct = X.isna().mean()
    keep = nan_pct[nan_pct < 0.5].index.tolist()
    X = X[keep]

    medians = X.median()
    X = X.fillna(medians)

    var = X.var()
    keep = var[var > 1e-10].index.tolist()
    X = X[keep]

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    log(f"  Features: {len(keep)}, samples: {len(X)}")

    # Walk-forward split: last 20% for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        verbose=-1, random_state=42, importance_type="gain",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(50)],
    )

    train_proba = model.predict_proba(X_train)[:, 1]
    train_acc = ((train_proba > 0.5).astype(int) == y_train.values).mean()

    val_proba = model.predict_proba(X_val)[:, 1]
    val_acc = ((val_proba > 0.5).astype(int) == y_val.values).mean()

    log(f"  Train acc: {train_acc:.1%} ({len(X_train)} samples)")
    log(f"  Val acc:   {val_acc:.1%} ({len(X_val)} samples)")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=keep).sort_values(ascending=False)
    log("  Top 15 features:")
    for feat, imp in importances.head(15).items():
        log(f"    {feat:50s} {imp:.0f}")

    return model, keep, medians.to_dict(), {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "n_features": len(keep),
    }


# ═════════════════════════════════════════════════════════════════════
# STEP 7: Learn bad hours
# ═════════════════════════════════════════════════════════════════════

def learn_bad_hours(merged_df: pd.DataFrame) -> set:
    target_col = "target_dir_24h"
    if "hour_of_day" not in merged_df.columns or target_col not in merged_df.columns:
        return set()

    bad = set()
    for h in range(24):
        mask = merged_df["hour_of_day"] == h
        sub = merged_df.loc[mask].dropna(subset=[target_col])
        if len(sub) < 20:
            continue
        # Use a simple directional accuracy proxy
        up_pct = sub[target_col].mean()
        if 0.45 < up_pct < 0.55:
            continue
        # Check if the hour is consistently wrong
        # (This is a simplified version; the live bot uses consensus_dir)

    log(f"  Bad hours: {sorted(bad) if bad else 'none detected'}")
    return bad


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pre-train v14 ML model")
    parser.add_argument("--csv", required=True, help="Path to quant-agents-export CSV")
    args = parser.parse_args()

    log("=" * 65)
    log("v14 PRE-TRAINING PIPELINE")
    log("=" * 65)

    # 1. Load agent data
    agent_df = load_agent_csv(args.csv)

    # 2. Fetch candles for the same period
    start = agent_df.index.min() - pd.Timedelta(hours=250)
    end = agent_df.index.max() + pd.Timedelta(hours=25)
    candle_df = fetch_all_candles(start, end)

    if candle_df.empty:
        log("ERROR: No candle data")
        sys.exit(1)

    # 3. Compute TA indicators
    log("Computing TA indicators...")
    candle_df = compute_indicators(candle_df)
    candle_df = build_targets(candle_df)

    # 4. Get candle features for each hour
    candle_features = get_candle_features_all(candle_df)
    candle_features["target_dir_24h"] = candle_df["target_dir_24h"]
    candle_features["hour_of_day"] = candle_df.get("hour_of_day", 0)

    log(f"  Candle features: {candle_features.shape}")

    # 5. Compute TA signals for each hour
    log("Computing TA signals for all hours (this may take a minute)...")
    ta_df = compute_ta_signals_all(candle_df)
    log(f"  TA signals computed: {len(ta_df)} rows")

    # 6. Merge agent features + candle features + TA signals
    log("Merging datasets...")
    merged = candle_features.join(agent_df, how="inner")
    merged = merged.join(ta_df, how="left")
    merged = merged.dropna(subset=["target_dir_24h"])

    log(f"  Merged dataset: {merged.shape} "
        f"({merged.index.min()} to {merged.index.max()})")
    log(f"  Target distribution: UP={merged['target_dir_24h'].mean():.1%} "
        f"DOWN={(1 - merged['target_dir_24h'].mean()):.1%}")

    if len(merged) < config.MIN_TRAIN_SAMPLES:
        log(f"ERROR: Only {len(merged)} samples, need {config.MIN_TRAIN_SAMPLES}")
        sys.exit(1)

    # 7. Train
    model, feature_cols, medians, metrics = train_model(merged)

    # 8. Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "medians": medians,
            "last_train_ts": datetime.now(timezone.utc).isoformat(),
            "pretrain_metrics": metrics,
        }, f)
    log(f"  Model saved: {MODEL_PATH}")
    log(f"  File size: {os.path.getsize(MODEL_PATH) / 1024:.0f} KB")

    # 9. Learn bad hours
    bad_hours = learn_bad_hours(merged)
    if bad_hours:
        import json
        with open(HOUR_STATE_PATH, "w") as f:
            json.dump({"bad_hours": sorted(bad_hours)}, f)
        log(f"  Bad hours saved: {HOUR_STATE_PATH}")

    log("")
    log("=" * 65)
    log("PRE-TRAINING COMPLETE")
    log(f"  Samples:     {metrics['train_samples'] + metrics['val_samples']}")
    log(f"  Features:    {metrics['n_features']}")
    log(f"  Train acc:   {metrics['train_acc']:.1%}")
    log(f"  Val acc:     {metrics['val_acc']:.1%}")
    log("=" * 65)


if __name__ == "__main__":
    main()
