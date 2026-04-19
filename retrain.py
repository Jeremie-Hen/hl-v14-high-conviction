#!/usr/bin/env python3
"""
Daily retrain — merges the original CSV with live-accumulated feature_store.jsonl,
refetches Binance candles up to yesterday, retrains the LightGBM model, and
refreshes bad_agents.json.

Called either:
  - internally by bot.py once per day at MODEL_RETRAIN_HOUR (default)
  - externally via a Railway cron service (set ENABLE_INTERNAL_RETRAIN=false
    on the bot service and set up a second Railway service that runs this
    script on a schedule)

Usage:
  python retrain.py
  python retrain.py --csv /path/to/initial/quant-agents-export.csv
"""
import argparse
import glob
import json
import os
import pickle
import re
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

import feature_store
from pretrain import (
    MODEL_PATH,
    BAD_AGENTS_PATH,
    DIRECTION_MAP,
    FORBIDDEN_FEATURES,
    load_agent_csv,
    fetch_all_candles,
    compute_indicators,
    compute_ta_signals_all,
    build_targets,
    get_candle_features_all,
    compute_agent_24h_accuracy,
    train_model,
)


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return re.sub(r"_+", "_", safe).strip("_")


def feature_store_to_agent_df() -> pd.DataFrame | None:
    """Pivot the JSONL feature store into the same shape as load_agent_csv's
    output (hour_key index, agent_{name}_dir columns)."""
    rows = feature_store.read_all()
    if not rows:
        return None

    log(f"  FeatureStore rows: {len(rows)}")

    # per-agent accuracy is computed later from forward candles; for the
    # training matrix we just need the _dir column per hour.
    seen_agents = {}  # raw_name -> safe_name
    out_rows = []
    for r in rows:
        hour_ts_str = r.get("hour_ts")
        if not hour_ts_str:
            continue
        try:
            hour_key = pd.to_datetime(hour_ts_str, utc=True).floor("h")
        except Exception:
            continue

        row = {"hour_key": hour_key}
        for a in r.get("agents", []):
            name = a.get("name")
            if not name:
                continue
            d = a.get("direction")
            if d is None or d == 0:
                continue
            safe = seen_agents.setdefault(name, _safe_name(name))
            row[f"agent_{safe}_dir"] = int(d)
        out_rows.append(row)

    if not out_rows:
        return None

    df = pd.DataFrame(out_rows).set_index("hour_key").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    log(f"  FeatureStore → agent matrix: {df.shape}")
    return df


def find_csv(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    # Look for any bundled CSV in the repo directory
    here = os.path.dirname(__file__)
    for path in glob.glob(os.path.join(here, "quant-agents-export*.csv")):
        return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Daily retrain of v14 ML model")
    parser.add_argument("--csv", help="Initial CSV (optional; auto-detects quant-agents-export*.csv in repo)")
    args = parser.parse_args()

    log("=" * 65)
    log("v14 DAILY RETRAIN")
    log("=" * 65)

    csv_path = find_csv(args.csv)
    base_agent_df = None
    if csv_path:
        log(f"Using initial CSV: {csv_path}")
        base_agent_df = load_agent_csv(csv_path)

    live_agent_df = feature_store_to_agent_df()

    if base_agent_df is None and live_agent_df is None:
        log("ERROR: no training data — provide --csv or accumulate feature_store rows")
        sys.exit(1)

    if base_agent_df is not None and live_agent_df is not None:
        combined = pd.concat([base_agent_df, live_agent_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        log(f"Combined agent matrix: {combined.shape} "
            f"({combined.index.min()} → {combined.index.max()})")
    elif live_agent_df is not None:
        combined = live_agent_df
    else:
        combined = base_agent_df

    start = combined.index.min() - pd.Timedelta(hours=250)
    end = combined.index.max() + pd.Timedelta(hours=25)
    candle_df = fetch_all_candles(start, end)
    if candle_df.empty:
        log("ERROR: no candle data")
        sys.exit(1)

    log("Computing TA indicators...")
    candle_df = compute_indicators(candle_df)
    candle_df = build_targets(candle_df)

    candle_features = get_candle_features_all(candle_df)
    candle_features["target_dir_24h"] = candle_df["target_dir_24h"]
    candle_features["hour_of_day"] = candle_df.get("hour_of_day", 0)

    log("Computing TA signals for all hours...")
    ta_df = compute_ta_signals_all(candle_df)

    merged = candle_features.join(combined, how="inner")
    merged = merged.join(ta_df, how="left")
    merged = merged.dropna(subset=["target_dir_24h"])
    log(f"Merged dataset: {merged.shape} "
        f"({merged.index.min()} → {merged.index.max()})")

    if len(merged) < config.MIN_TRAIN_SAMPLES:
        log(f"ERROR: only {len(merged)} samples, need {config.MIN_TRAIN_SAMPLES}")
        sys.exit(1)

    model, feature_cols, medians, metrics = train_model(merged)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "medians": medians,
            "last_train_ts": datetime.now(timezone.utc).isoformat(),
            "pretrain_metrics": metrics,
        }, f)
    log(f"  Model saved: {MODEL_PATH}")

    # Recompute bad-agents stats. Prefer the original CSV if we still have it
    # (far more rows than live accumulation would have); otherwise fall back
    # to the feature-store-only path (future-proofing).
    if csv_path:
        agent_stats = compute_agent_24h_accuracy(csv_path, candle_df)
        bad_list = sorted([
            n for n, d in agent_stats.items()
            if d["lifetime_acc_24h"] <= config.BAD_AGENT_THRESHOLD
            and d["total_predictions"] >= config.BAD_AGENT_MIN_HISTORY
        ])
        with open(BAD_AGENTS_PATH, "w") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "threshold": config.BAD_AGENT_THRESHOLD,
                "min_history": config.BAD_AGENT_MIN_HISTORY,
                "horizon": "24h",
                "agents": agent_stats,
                "bad_agents": bad_list,
            }, f, indent=2)
        log(f"  Bad-agents refreshed: {len(bad_list)} bad / {len(agent_stats)} total")

    log("=" * 65)
    log(f"RETRAIN COMPLETE — train={metrics['train_acc']:.1%} val={metrics['val_acc']:.1%} "
        f"n_features={metrics['n_features']} samples={metrics['train_samples'] + metrics['val_samples']}")
    log("=" * 65)


if __name__ == "__main__":
    main()
