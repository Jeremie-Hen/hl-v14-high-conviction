"""
Technical Analysis Engine — v14 High-Conviction Bot
Computes 28 TA strategy signals on 1h candles, outputs ta_net_signal in [-1,1].

Strategies: MA crosses (multiple periods), RSI, MACD, Bollinger Bands,
Ichimoku, ADX/DMI, Donchian channels.
"""
import numpy as np
import pandas as pd
import requests

import config


def log(msg):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def fetch_binance_candles(interval: str = "1h", limit: int = 300) -> pd.DataFrame:
    """Fetch BTCUSDT candles from Binance REST API."""
    url = f"{config.BINANCE_REST_URL}/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 2:
                log(f"  Binance fetch error (attempt {attempt+1}): {e}")
                return pd.DataFrame()
            import time; time.sleep(2)

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("open_time").sort_index()
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the candle DataFrame."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Moving averages ──────────────────────────────────────────
    for span in [9, 12, 21, 26, 50, 100, 200]:
        df[f"ema_{span}"] = c.ewm(span=span).mean()
    for window in [20, 50, 100, 200]:
        df[f"sma_{window}"] = c.rolling(window).mean()

    # ── Returns ──────────────────────────────────────────────────
    df["return_1h"] = c.pct_change()
    df["return_4h"] = c.pct_change(4)
    df["return_24h"] = c.pct_change(24)

    # ── ATR ──────────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / c * 100

    # ── Volatility ───────────────────────────────────────────────
    df["vol_24h"] = df["return_1h"].rolling(24).std()
    df["vol_72h"] = df["return_1h"].rolling(72).std()
    df["vol_ratio"] = df["vol_24h"] / df["vol_72h"]

    # ── Trend EMA ────────────────────────────────────────────────
    df["trend_ema"] = (df["ema_12"] - df["ema_26"]) / c * 100

    # ── Bollinger Bands ──────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_bandwidth"] = (2 * bb_std / bb_mid * 100)
    df["bb_pct_b"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── RSI ──────────────────────────────────────────────────────
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────
    df["macd"] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── ADX / DMI ────────────────────────────────────────────────
    plus_dm = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_14 = tr.rolling(14).mean()
    df["plus_di"] = 100 * (plus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
    df["minus_di"] = 100 * (minus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
    dx = (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"]).replace(0, np.nan) * 100
    df["adx_14"] = dx.rolling(14).mean()

    # ── Ichimoku ─────────────────────────────────────────────────
    df["ichi_tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    df["ichi_kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    df["ichi_senkou_a"] = ((df["ichi_tenkan"] + df["ichi_kijun"]) / 2).shift(26)
    df["ichi_senkou_b"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)

    # ── Donchian Channel ─────────────────────────────────────────
    df["donchian_upper_20"] = h.rolling(20).max()
    df["donchian_lower_20"] = l.rolling(20).min()
    df["donchian_mid_20"] = (df["donchian_upper_20"] + df["donchian_lower_20"]) / 2

    # ── Volume ───────────────────────────────────────────────────
    df["vol_sma_24"] = v.rolling(24).mean()
    df["vol_ratio_24"] = v / df["vol_sma_24"]

    # ── Regime ───────────────────────────────────────────────────
    df["regime"] = "range"
    df.loc[(df["adx_14"] > 25) & (df["trend_ema"] > 0), "regime"] = "trend_up"
    df.loc[(df["adx_14"] > 25) & (df["trend_ema"] < 0), "regime"] = "trend_down"
    df.loc[df["vol_ratio"] > 1.5, "regime"] = "high_vol"

    # ── Hour of day ──────────────────────────────────────────────
    if hasattr(df.index, "hour"):
        df["hour_of_day"] = df.index.hour
    else:
        df["hour_of_day"] = 0

    return df


def compute_ta_signals(df: pd.DataFrame) -> dict:
    """
    Evaluate 28 TA strategies on the latest row and return a dict with
    ta_net_signal (normalized to [-1,1]) and individual signal details.
    """
    if len(df) < 200:
        return {"ta_net_signal": 0.0, "ta_long": 0, "ta_short": 0, "signals": {}}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    signals = {}

    def _signal(name, is_long, is_short):
        if is_long:
            signals[name] = 1
        elif is_short:
            signals[name] = -1
        else:
            signals[name] = 0

    # 1-8: EMA crosses
    ema_pairs = [(9, 21), (9, 26), (12, 26), (12, 50), (21, 50), (26, 50), (50, 100), (50, 200)]
    for fast, slow in ema_pairs:
        fc, sc = f"ema_{fast}", f"ema_{slow}"
        if fc in df.columns and sc in df.columns:
            cross_up = prev[fc] <= prev[sc] and last[fc] > last[sc]
            cross_dn = prev[fc] >= prev[sc] and last[fc] < last[sc]
            above = last[fc] > last[sc]
            _signal(f"ema_{fast}_{slow}", cross_up or above, cross_dn or (not above))

    # 9-10: SMA crosses
    for fast, slow in [(50, 200), (20, 50)]:
        fc, sc = f"sma_{fast}", f"sma_{slow}"
        if fc in df.columns and sc in df.columns:
            _signal(f"sma_{fast}_{slow}", last[fc] > last[sc], last[fc] < last[sc])

    # 11: RSI
    if "rsi_14" in df.columns and not np.isnan(last["rsi_14"]):
        _signal("rsi_14", last["rsi_14"] < 30, last["rsi_14"] > 70)

    # 12: RSI midline cross
    if "rsi_14" in df.columns and not np.isnan(last["rsi_14"]):
        _signal("rsi_midline", last["rsi_14"] > 50 and prev.get("rsi_14", 50) <= 50,
                last["rsi_14"] < 50 and prev.get("rsi_14", 50) >= 50)

    # 13: MACD histogram
    if "macd_hist" in df.columns and not np.isnan(last["macd_hist"]):
        _signal("macd_hist", last["macd_hist"] > 0, last["macd_hist"] < 0)

    # 14: MACD cross
    if "macd" in df.columns and "macd_signal" in df.columns:
        _signal("macd_cross",
                prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"],
                prev["macd"] >= prev["macd_signal"] and last["macd"] < last["macd_signal"])

    # 15: Bollinger %B
    if "bb_pct_b" in df.columns and not np.isnan(last["bb_pct_b"]):
        _signal("bb_pct_b", last["bb_pct_b"] < 0.0, last["bb_pct_b"] > 1.0)

    # 16: Bollinger squeeze breakout
    if "bb_bandwidth" in df.columns:
        bw = df["bb_bandwidth"].dropna()
        if len(bw) > 20:
            low_bw = bw.quantile(0.1)
            is_squeeze = prev.get("bb_bandwidth", 999) < low_bw and last["bb_bandwidth"] > low_bw
            _signal("bb_squeeze", is_squeeze and last.get("close", 0) > last.get("bb_upper", 0),
                    is_squeeze and last.get("close", 0) < last.get("bb_lower", 0))

    # 17: ADX trend strength
    if "adx_14" in df.columns and "plus_di" in df.columns:
        strong_trend = last.get("adx_14", 0) > 25
        _signal("adx_trend", strong_trend and last["plus_di"] > last["minus_di"],
                strong_trend and last["minus_di"] > last["plus_di"])

    # 18: ADX rising
    if "adx_14" in df.columns:
        _signal("adx_rising", last["adx_14"] > prev.get("adx_14", 0) and last.get("plus_di", 0) > last.get("minus_di", 0),
                last["adx_14"] > prev.get("adx_14", 0) and last.get("minus_di", 0) > last.get("plus_di", 0))

    # 19: Ichimoku cloud position
    if all(c in df.columns for c in ["ichi_senkou_a", "ichi_senkou_b"]):
        cloud_top = max(last.get("ichi_senkou_a", 0), last.get("ichi_senkou_b", 0))
        cloud_bot = min(last.get("ichi_senkou_a", 0), last.get("ichi_senkou_b", 0))
        _signal("ichi_cloud", last["close"] > cloud_top, last["close"] < cloud_bot)

    # 20: Ichimoku TK cross
    if "ichi_tenkan" in df.columns and "ichi_kijun" in df.columns:
        _signal("ichi_tk_cross",
                prev.get("ichi_tenkan", 0) <= prev.get("ichi_kijun", 0) and last["ichi_tenkan"] > last["ichi_kijun"],
                prev.get("ichi_tenkan", 0) >= prev.get("ichi_kijun", 0) and last["ichi_tenkan"] < last["ichi_kijun"])

    # 21: Ichimoku price vs kijun
    if "ichi_kijun" in df.columns:
        _signal("ichi_kijun", last["close"] > last["ichi_kijun"], last["close"] < last["ichi_kijun"])

    # 22: Donchian breakout
    if "donchian_upper_20" in df.columns:
        _signal("donchian_break",
                last["close"] >= last["donchian_upper_20"],
                last["close"] <= last["donchian_lower_20"])

    # 23: Donchian mid reversion
    if "donchian_mid_20" in df.columns:
        _signal("donchian_mid",
                last["close"] > last["donchian_mid_20"] and prev.get("close", 0) <= prev.get("donchian_mid_20", 0),
                last["close"] < last["donchian_mid_20"] and prev.get("close", 0) >= prev.get("donchian_mid_20", 0))

    # 24: Price vs EMA 200
    if "ema_200" in df.columns:
        _signal("price_vs_ema200", last["close"] > last["ema_200"], last["close"] < last["ema_200"])

    # 25: Price vs SMA 200
    if "sma_200" in df.columns:
        _signal("price_vs_sma200", last["close"] > last["sma_200"], last["close"] < last["sma_200"])

    # 26: Volume spike + trend
    if "vol_ratio_24" in df.columns and "trend_ema" in df.columns:
        vol_spike = last.get("vol_ratio_24", 1.0) > 1.5
        _signal("vol_spike_trend",
                vol_spike and last.get("trend_ema", 0) > 0,
                vol_spike and last.get("trend_ema", 0) < 0)

    # 27: Mean reversion (RSI + BB)
    if "rsi_14" in df.columns and "bb_pct_b" in df.columns:
        _signal("mean_revert",
                last.get("rsi_14", 50) < 35 and last.get("bb_pct_b", 0.5) < 0.1,
                last.get("rsi_14", 50) > 65 and last.get("bb_pct_b", 0.5) > 0.9)

    # 28: Multi-timeframe trend agreement (use multiple EMA slopes)
    if all(f"ema_{s}" in df.columns for s in [12, 26, 50]):
        all_up = last["ema_12"] > last["ema_26"] > last["ema_50"]
        all_dn = last["ema_12"] < last["ema_26"] < last["ema_50"]
        _signal("multi_ema_trend", all_up, all_dn)

    # Aggregate
    ta_long = sum(1 for v in signals.values() if v == 1)
    ta_short = sum(1 for v in signals.values() if v == -1)
    total = len(signals)

    ta_net = (ta_long - ta_short) / total if total > 0 else 0.0
    ta_net = max(-1.0, min(1.0, ta_net))

    return {
        "ta_net_signal": ta_net,
        "ta_long": ta_long,
        "ta_short": ta_short,
        "total_strategies": total,
        "signals": signals,
    }


def get_candle_features(df: pd.DataFrame) -> dict:
    """Extract the latest row's numeric features for the ML model."""
    if df.empty:
        return {}
    last = df.iloc[-1]
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
    features = {}
    for col in feature_cols:
        if col in df.columns:
            val = last[col]
            if isinstance(val, (int, float, np.integer, np.floating)) and not np.isnan(val):
                features[col] = float(val)
    return features
