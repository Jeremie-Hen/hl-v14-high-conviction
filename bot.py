#!/usr/bin/env python3
"""
v14 24h High-Conviction Meta-Strategy Bot
==========================================
Hourly loop: poll myquant agents, fetch 1h candles, compute 4-component
conviction score, execute on Hyperliquid at 10x leverage with dynamic
ATR-based trailing stop.

Execution Flow (each hour):
  1. Wait for hour boundary
  2. Wait ~90s for myquant predictions to generate, poll for fresh data
  3. Fetch 300 1h candles from Binance
  4. Build features (agent + TA + price)
  5. Compute conviction score (ML + counter + TA + consensus)
  6. Manage existing position (trailing stop, max hold, reversal)
  7. Open new position if conviction >= threshold
  8. Between hours: poll price every 60s for trailing stop management
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import config
from hyperliquid_client import HyperliquidClient
from myquant_client import (
    AgentHistoryTracker,
    build_fingerprint,
    poll_for_fresh_predictions,
)
from ta_engine import fetch_binance_candles, compute_indicators, compute_ta_signals, get_candle_features
from signal_engine import MLModel, HourFilter, compute_signal
from performance_tracker import PerformanceTracker
import feature_store

BOT_STATE_PATH = os.path.join(os.path.dirname(__file__), "bot_state.json")
RETRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "retrain.py")


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def save_bot_state(state: dict):
    try:
        with open(BOT_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log(f"  [STATE] Error saving: {e}")


def load_bot_state() -> dict:
    if os.path.exists(BOT_STATE_PATH):
        try:
            with open(BOT_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_fingerprint": None, "last_processed_hour": None}


# ═════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT
# ═════════════════════════════════════════════════════════════════════

def compute_position_size(equity: float, price: float) -> tuple[float, float]:
    """Returns (size_btc, size_usd). Full equity * leverage compounding."""
    size_usd = equity * config.POSITION_EQUITY_PCT * config.LEVERAGE
    size_btc = size_usd / price
    return round(size_btc, 5), round(size_usd, 2)


def open_new_position(hl: HyperliquidClient, tracker: PerformanceTracker, signal: dict, equity: float) -> bool:
    """Open a position on Hyperliquid based on the signal."""
    mid_price = hl.get_mid_price(config.SYMBOL)
    size_btc, size_usd = compute_position_size(equity, mid_price)

    if size_btc < 0.001:
        log(f"  [POS] Size too small: {size_btc} BTC (${size_usd})")
        return False

    is_long = signal["direction"] == "LONG"
    log(f"  [POS] Opening {signal['direction']} {size_btc} BTC (${size_usd:.2f}) "
        f"@ ${mid_price:,.0f} | {config.LEVERAGE}x | "
        f"SL={signal['sl_pct']:.2f}%")

    tracker.open_position(signal, mid_price, size_btc, size_usd)

    if config.DRY_RUN:
        log(f"  [POS] DRY RUN — order simulated")
        return True

    try:
        hl.set_leverage(config.SYMBOL, config.LEVERAGE, is_cross=True)
        time.sleep(0.3)

        result = hl.place_market_order(config.SYMBOL, is_buy=is_long, size=size_btc)
        if isinstance(result, dict) and result.get("status") == "error":
            log(f"  [POS] Entry FAILED: {result}")
            tracker.open_trade = None
            tracker.save()
            return False

        fill_price = HyperliquidClient.extract_fill_price(result)
        if fill_price is not None:
            log(f"  [POS] Fill price: ${fill_price:,.2f} (mid was ${mid_price:,.0f})")
            tracker.open_trade["entry_price"] = fill_price
            sl_pct = signal["sl_pct"]
            if is_long:
                tracker.open_trade["sl_price"] = fill_price * (1 - sl_pct / 100)
            else:
                tracker.open_trade["sl_price"] = fill_price * (1 + sl_pct / 100)

        time.sleep(0.5)

        sl_price = tracker.open_trade["sl_price"]
        sl_result = hl.place_sl_order(config.SYMBOL, size_btc, is_long, sl_price)
        sl_oid = sl_result.get("sl_oid")
        tracker.open_trade["sl_oid"] = sl_oid
        log(f"  [POS] SL placed: oid={sl_oid} @ ${sl_price:,.0f}")
        tracker.save()

        return True
    except Exception as e:
        log(f"  [POS] Error opening: {e}")
        traceback.print_exc()
        return False


def close_existing_position(hl: HyperliquidClient, tracker: PerformanceTracker, reason: str) -> bool:
    """Close the current open position. Cancels SL by OID then market-closes."""
    if not tracker.has_open_position:
        return False

    log(f"  [POS] Closing position: {reason}")

    sl_oid = tracker.open_trade.get("sl_oid") if tracker.open_trade else None
    if sl_oid and not config.DRY_RUN:
        try:
            hl.cancel_orders(config.SYMBOL, [sl_oid])
            log(f"  [POS] Cancelled SL oid={sl_oid}")
        except Exception as e:
            log(f"  [POS] Error cancelling SL: {e}")

    mid_price = hl.get_mid_price(config.SYMBOL)

    if not config.DRY_RUN:
        try:
            positions = hl.get_open_positions()
            for pos in positions:
                if pos["symbol"] == config.SYMBOL:
                    is_buy = pos["size"] < 0
                    hl.place_market_order(config.SYMBOL, is_buy, abs(pos["size"]),
                                          reduce_only=True)
                    break
        except Exception as e:
            log(f"  [POS] Error closing on HL: {e}")

    tracker.close_position(mid_price, reason)
    return True


def manage_position_between_hours(hl: HyperliquidClient, tracker: PerformanceTracker):
    """
    Between hourly cycles: poll price, update trailing stop, check SL hit.
    Runs every POSITION_POLL_SEC.
    """
    if not tracker.has_open_position:
        return

    try:
        mid_price = hl.get_mid_price(config.SYMBOL)
    except Exception:
        return

    trade = tracker.open_trade
    old_sl = tracker.get_effective_sl()

    tracker.update_trailing(mid_price)

    new_sl = tracker.get_effective_sl()

    if trade.get("trail_active") and new_sl and old_sl and new_sl != old_sl:
        is_long = trade["direction"] == "LONG"
        old_sl_oid = trade.get("sl_oid")
        if not config.DRY_RUN:
            try:
                sl_result = hl.update_trailing_stop(
                    config.SYMBOL, trade["size_btc"], is_long, new_sl,
                    old_sl_oid=old_sl_oid,
                )
                new_oid = sl_result.get("sl_oid")
                trade["sl_oid"] = new_oid
                tracker.save()
                log(f"  [TRAIL] Updated SL: ${old_sl:,.0f} -> ${new_sl:,.0f} (oid={new_oid})")
            except Exception as e:
                log(f"  [TRAIL] Error updating: {e}")
        else:
            log(f"  [TRAIL] DRY RUN SL update: ${old_sl:,.0f} -> ${new_sl:,.0f}")

    if not config.DRY_RUN:
        reconcile_position_with_exchange(hl, tracker, mid_price)


def reconcile_position_with_exchange(hl: HyperliquidClient, tracker: PerformanceTracker, mid_price: float):
    """
    OID-based reconciliation: check if the SL oid is still alive.
    If it's gone, the SL was triggered on-chain — close the tracker.
    Also verify exchange position still exists as a safety check.
    """
    if not tracker.has_open_position:
        return

    trade = tracker.open_trade
    sl_oid = trade.get("sl_oid")

    if sl_oid:
        try:
            open_orders = hl.get_open_orders()
            live_oids = {int(o["oid"]) for o in open_orders if "oid" in o}

            if int(sl_oid) not in live_oids:
                log(f"  [RECONCILE] SL oid={sl_oid} no longer alive — SL was triggered")
                tracker.close_position(mid_price, "sl_triggered")
                return
        except Exception:
            pass

    try:
        positions = hl.get_open_positions()
        has_pos = any(p["symbol"] == config.SYMBOL for p in positions)
        if not has_pos and tracker.has_open_position:
            log("  [RECONCILE] No exchange position found — closed externally")
            tracker.close_position(mid_price, "external_close")
    except Exception:
        pass


def check_position_expiry(hl: HyperliquidClient, tracker: PerformanceTracker) -> bool:
    """Check if max hold time exceeded. Returns True if position was closed."""
    if not tracker.has_open_position:
        return False

    hours = tracker.open_trade.get("hours_held", 0)
    if hours >= config.MAX_HOLD_HOURS:
        close_existing_position(hl, tracker, f"max_hold_{hours}h")
        return True
    return False


def maybe_retrain_model(bot_state: dict, ml_model: MLModel, agent_tracker) -> None:
    """Spawn retrain.py once per day at config.MODEL_RETRAIN_HOUR.
    The subprocess rewrites model_v14.pkl + bad_agents.json on disk, which
    both the ML model and agent tracker reload on next use."""
    if not config.ENABLE_INTERNAL_RETRAIN:
        return
    if not os.path.exists(RETRAIN_SCRIPT):
        return

    now = datetime.now(timezone.utc)
    if now.hour != config.MODEL_RETRAIN_HOUR:
        return

    today_key = now.strftime("%Y-%m-%d")
    if bot_state.get("last_retrain_day") == today_key:
        return

    log(f"  [RETRAIN] Triggering daily retrain ({today_key}, hour {now.hour}:00 UTC)...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, RETRAIN_SCRIPT],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=1200,
        )
        bot_state["last_retrain_day"] = today_key
        if result.returncode == 0:
            log("  [RETRAIN] Succeeded — reloading model & agent stats")
            tail = (result.stdout or "").strip().splitlines()[-10:]
            for line in tail:
                log(f"    {line}")
            ml_model._load()
            agent_tracker._load_precomputed()
        else:
            log(f"  [RETRAIN] FAILED (exit {result.returncode})")
            err_tail = (result.stderr or "").strip().splitlines()[-15:]
            for line in err_tail:
                log(f"    {line}")
    except Exception as e:
        log(f"  [RETRAIN] Error spawning: {e}")


def check_signal_reversal(
    hl: HyperliquidClient,
    tracker: PerformanceTracker,
    new_signal: dict | None,
) -> bool:
    """
    If a new signal contradicts the open position, close and reverse.
    Returns True if reversed.
    """
    if not tracker.has_open_position or not new_signal:
        return False

    current_dir = tracker.open_trade["direction"]
    new_dir = new_signal["direction"]

    if current_dir == new_dir:
        return False

    log(f"  [REVERSAL] {current_dir} -> {new_dir} (conviction={new_signal['conviction']:+.3f})")
    close_existing_position(hl, tracker, f"reversal_to_{new_dir}")
    return True


# ═════════════════════════════════════════════════════════════════════
# MAIN BOT LOOP
# ═════════════════════════════════════════════════════════════════════

def main():
    log("=" * 65)
    log("v14 24h HIGH-CONVICTION META-STRATEGY BOT")
    log("=" * 65)
    log(f"Symbol:        {config.SYMBOL}")
    log(f"Leverage:      {config.LEVERAGE}x")
    log(f"Conviction:    >= {config.CONVICTION_THRESHOLD}")
    log(f"Max hold:      {config.MAX_HOLD_HOURS}h")
    log(f"Trail:         activate={config.TRAIL_ACTIVATE_PCT}% distance={config.TRAIL_DISTANCE_PCT}%")
    log(f"SL:            ATR*{config.SL_MULTIPLIER} (clamped {config.SL_MIN_PCT}-{config.SL_MAX_PCT}%)")
    log(f"Weights:       ML={config.W_ML} CT={config.W_COUNTER} TA={config.W_TA} CS={config.W_CONSENSUS}")
    log(f"Live trading:  {config.LIVE_TRADING}")
    log(f"Dry run:       {config.DRY_RUN}")
    log(f"Testnet:       {config.TESTNET}")
    log(f"API:           {config.API_URL}")
    log("")

    # Initialize components
    ml_model = MLModel()
    hour_filter = HourFilter()
    agent_tracker = AgentHistoryTracker()
    bot_state = load_bot_state()

    # Initialize Hyperliquid client
    hl = None
    initial_equity = 100.0

    if config.LIVE_TRADING and not config.DRY_RUN:
        try:
            hl = HyperliquidClient()
            initial_equity = hl.get_account_value()
            if initial_equity <= 0:
                initial_equity = 100.0
                log("  WARNING: Could not get account value, using $100 placeholder")
        except Exception as e:
            log(f"  HL client error: {e}. Running in DRY RUN mode.")
            config.DRY_RUN = True
    else:
        log("  Running in DRY RUN mode")

    if hl is None:
        hl = HyperliquidClient()

    tracker = PerformanceTracker(initial_equity=initial_equity)
    if tracker.current_equity <= 0:
        tracker.current_equity = initial_equity
        tracker.peak_equity = initial_equity

    report_counter = 0
    last_fingerprint = bot_state.get("last_fingerprint")

    log(f"  Equity: ${tracker.current_equity:.2f}")
    log(f"  Open position: {'YES' if tracker.has_open_position else 'NO'}")

    # Startup reconciliation: verify position still exists on exchange
    if config.LIVE_TRADING and not config.DRY_RUN and tracker.has_open_position:
        log("  [STARTUP] Reconciling with exchange...")
        try:
            mid = hl.get_mid_price(config.SYMBOL)
            reconcile_position_with_exchange(hl, tracker, mid)
        except Exception as e:
            log(f"  [STARTUP] Reconciliation error: {e}")

    # Cleanup orphaned trigger orders from previous runs
    if config.LIVE_TRADING and not config.DRY_RUN:
        log("  [STARTUP] Checking for orphaned orders...")
        tracked_oids = set()
        if tracker.has_open_position and tracker.open_trade:
            sl_oid = tracker.open_trade.get("sl_oid")
            if sl_oid:
                tracked_oids.add(int(sl_oid))
        orphans = hl.cleanup_orphaned_orders(config.SYMBOL, tracked_oids)
        if orphans:
            log(f"  [STARTUP] Cleaned {orphans} orphaned orders")
        else:
            log("  [STARTUP] No orphaned orders")

    log("")
    log("Bot started. Entering hourly loop...")
    log("")

    while True:
        try:
            tracker.check_daily_reset()
            maybe_retrain_model(bot_state, ml_model, agent_tracker)

            # ── Wait for next hour boundary ──────────────────────
            now = datetime.now(timezone.utc)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            wait_sec = (next_hour - now).total_seconds()

            if wait_sec > 120:
                log(f"Waiting {wait_sec:.0f}s for next hour ({next_hour.strftime('%H:%M')} UTC)...")

                poll_deadline = time.time() + wait_sec - 90
                while time.time() < poll_deadline:
                    manage_position_between_hours(hl, tracker)
                    sleep_time = min(config.POSITION_POLL_SEC, poll_deadline - time.time())
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                remaining = (next_hour - datetime.now(timezone.utc)).total_seconds()
                if remaining > 0:
                    time.sleep(max(remaining - 5, 0))

            # ── Small buffer past the hour for data finalization ─
            time.sleep(5)

            now = datetime.now(timezone.utc)
            current_hour_str = now.strftime("%Y-%m-%d %H:00")

            if bot_state.get("last_processed_hour") == current_hour_str:
                log(f"  Already processed {current_hour_str}, skipping")
                time.sleep(60)
                continue

            log("")
            log("=" * 60)
            log(f"HOUR: {now.strftime('%Y-%m-%d %H:%M')} UTC")
            log("=" * 60)

            # ── Poll myquant for fresh predictions ───────────────
            log("  Polling myquant.gg for fresh predictions...")
            agents, poll_stats = poll_for_fresh_predictions(last_fingerprint)

            if agents is None:
                log("  WARNING: No fresh predictions, proceeding with stale data")
                from myquant_client import fetch_predictions, parse_all_agents
                raw = fetch_predictions()
                if raw:
                    agents = parse_all_agents(raw)

            if agents:
                last_fingerprint = build_fingerprint(agents)
                agent_tracker.record_predictions(agents, current_hour_str)
                agent_tracker.save()
                agent_features = agent_tracker.build_agent_features(agents)
                log(f"  Agents: {len(agents)} predictions, "
                    f"{sum(1 for k in agent_features if k.endswith('_dir'))} with features")
            else:
                agent_features = {}
                log("  WARNING: No agent data available")

            # ── Fetch candles and compute TA ──────────────────────
            log("  Fetching 1h candles from Binance...")
            df_candles = fetch_binance_candles(interval="1h", limit=300)

            if df_candles.empty:
                log("  ERROR: No candle data")
                time.sleep(60)
                continue

            df_candles = compute_indicators(df_candles)
            ta_result = compute_ta_signals(df_candles)
            candle_features = get_candle_features(df_candles)
            btc_price = df_candles["close"].iloc[-1]
            atr_pct = candle_features.get("atr_pct", 0.5)

            log(f"  BTC: ${btc_price:,.0f} | ATR: {atr_pct:.2f}% | "
                f"TA: {ta_result['ta_long']}L/{ta_result['ta_short']}S "
                f"(net={ta_result['ta_net_signal']:+.3f})")

            # Persist this hour's snapshot so retrain.py can refresh the model
            feature_store.append_hour(
                hour_ts=current_hour_str,
                btc_close=btc_price,
                agents=agents or [],
                candle_features=candle_features,
                ta_result=ta_result,
            )

            # ── ML model status ────────────────────────────────────
            if ml_model.model is not None:
                age = ""
                if ml_model.last_train_ts:
                    try:
                        trained_dt = datetime.fromisoformat(ml_model.last_train_ts)
                        age_h = (now - trained_dt).total_seconds() / 3600
                        age = f", trained {age_h:.0f}h ago"
                    except Exception:
                        pass
                log(f"  [ML] Model loaded ({len(ml_model.feature_cols)} features{age})")
            else:
                log("  [ML] No model loaded — trading without ML component")

            # ── Compute signal ────────────────────────────────────
            tracker.signals_generated += 1
            signal = compute_signal(
                ml_model=ml_model,
                hour_filter=hour_filter,
                candle_features=candle_features,
                agent_features=agent_features,
                ta_result=ta_result,
                current_hour=now.hour,
                atr_pct=atr_pct,
            )

            # ── Position management ───────────────────────────────
            if tracker.has_open_position:
                tracker.increment_hold_hours()

                # 1. Check max hold
                if check_position_expiry(hl, tracker):
                    log("  Position closed: max hold time reached")

                # 2. Check signal reversal
                elif signal and check_signal_reversal(hl, tracker, signal):
                    open_new_position(hl, tracker, signal, tracker.current_equity)

                # 3. Update trailing stop + reconcile
                else:
                    manage_position_between_hours(hl, tracker)

            # ── Open new position if no position ──────────────────
            if not tracker.has_open_position and signal:
                open_new_position(hl, tracker, signal, tracker.current_equity)
            elif not signal:
                tracker.signals_filtered += 1

            # ── Update state ──────────────────────────────────────
            bot_state["last_fingerprint"] = last_fingerprint
            bot_state["last_processed_hour"] = current_hour_str
            save_bot_state(bot_state)

            # ── Status ────────────────────────────────────────────
            tracker.log_status()

            report_counter += 1
            if report_counter >= 24:
                tracker.log_report()
                report_counter = 0

        except KeyboardInterrupt:
            log("\nShutdown requested...")
            if tracker.has_open_position and not config.DRY_RUN:
                log("  Closing open position...")
                try:
                    close_existing_position(hl, tracker, "shutdown")
                except Exception as e:
                    log(f"  Error closing: {e}")
            tracker.log_report()
            break

        except Exception as e:
            log(f"ERROR in main loop: {e}")
            traceback.print_exc()
            time.sleep(30)

    log("Bot stopped.")


if __name__ == "__main__":
    main()
