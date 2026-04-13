"""
Performance Tracker — v14 High-Conviction Bot
Trade logging, equity curve tracking, drawdown, stats, crash recovery.
"""
import json
import os
from datetime import datetime, timezone

import config


def _ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "trade_log.json")
INITIAL_EQUITY = 100.0  # placeholder; overwritten from HL on boot


class PerformanceTracker:
    def __init__(self, initial_equity: float = INITIAL_EQUITY):
        self.trades: list[dict] = []
        self.open_trade: dict | None = None

        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.max_drawdown = 0.0

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_date = datetime.now(timezone.utc).date()

        self.signals_generated = 0
        self.signals_executed = 0
        self.signals_filtered = 0

        self._load()

    # ── Persistence ──────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(TRADE_LOG_PATH):
            return
        try:
            with open(TRADE_LOG_PATH, "r") as f:
                data = json.load(f)
            self.trades = data.get("trades", [])
            self.open_trade = data.get("open_trade")
            self.peak_equity = data.get("peak_equity", self.peak_equity)
            self.current_equity = data.get("current_equity", self.current_equity)
            self.max_drawdown = data.get("max_drawdown", 0.0)
            self.signals_generated = data.get("signals_generated", 0)
            self.signals_executed = data.get("signals_executed", 0)
            self.signals_filtered = data.get("signals_filtered", 0)
            n = len(self.trades)
            print(f"  [PERF] Loaded {n} trades, equity=${self.current_equity:.2f}, "
                  f"maxDD={self.max_drawdown:.1%}", flush=True)
        except Exception as e:
            print(f"  [PERF] Error loading: {e}", flush=True)

    def save(self):
        try:
            data = {
                "trades": self.trades[-500:],
                "open_trade": self.open_trade,
                "peak_equity": self.peak_equity,
                "current_equity": self.current_equity,
                "max_drawdown": self.max_drawdown,
                "signals_generated": self.signals_generated,
                "signals_executed": self.signals_executed,
                "signals_filtered": self.signals_filtered,
                "last_updated": _ts(),
            }
            with open(TRADE_LOG_PATH, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"  [PERF] Error saving: {e}", flush=True)

    # ── Trade lifecycle ──────────────────────────────────────────

    @property
    def has_open_position(self) -> bool:
        return self.open_trade is not None

    def open_position(
        self,
        signal: dict,
        entry_price: float,
        size_btc: float,
        size_usd: float,
    ):
        self.open_trade = {
            "direction": signal["direction"],
            "entry_price": entry_price,
            "size_btc": size_btc,
            "size_usd": size_usd,
            "leverage": signal["leverage"],
            "conviction": signal["conviction"],
            "sl_pct": signal["sl_pct"],
            "sl_price": entry_price * (1 - signal["sl_pct"] / 100) if signal["direction"] == "LONG"
                        else entry_price * (1 + signal["sl_pct"] / 100),
            "trail_active": False,
            "trail_sl_price": None,
            "max_favorable_price": entry_price,
            "entry_time": _ts(),
            "entry_hour": datetime.now(timezone.utc).hour,
            "hours_held": 0,
            "components": signal.get("components", {}),
        }
        self.signals_executed += 1
        self.save()

    def close_position(self, exit_price: float, outcome: str) -> dict | None:
        if not self.open_trade:
            return None

        trade = self.open_trade
        direction = trade["direction"]
        entry = trade["entry_price"]

        if direction == "LONG":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        fee_pct = config.TAKER_FEE_PCT * 2
        pnl_pct_after_fees = pnl_pct - fee_pct

        lev = trade["leverage"]
        equity_pnl_pct = pnl_pct_after_fees * lev
        pnl_usd = equity_pnl_pct * self.current_equity

        closed = {
            **trade,
            "exit_price": exit_price,
            "exit_time": _ts(),
            "outcome": outcome,
            "pnl_pct": pnl_pct_after_fees,
            "pnl_pct_leveraged": equity_pnl_pct,
            "pnl_usd": pnl_usd,
        }

        self.trades.append(closed)
        self.open_trade = None

        self.current_equity += pnl_usd
        self.peak_equity = max(self.peak_equity, self.current_equity)
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, dd)

        self.daily_pnl += pnl_usd
        self.daily_trades += 1
        self.save()

        icon = "WIN" if pnl_usd > 0 else "LOSS"
        print(f"  [TRADE] {icon}: {direction} | "
              f"${entry:,.0f} -> ${exit_price:,.0f} | "
              f"PnL: ${pnl_usd:.2f} ({equity_pnl_pct:+.2%}) | "
              f"Outcome: {outcome}", flush=True)

        return closed

    def increment_hold_hours(self):
        if self.open_trade:
            self.open_trade["hours_held"] = self.open_trade.get("hours_held", 0) + 1

    def update_trailing(self, current_price: float):
        """Update max favorable price and trailing stop if activated."""
        if not self.open_trade:
            return

        trade = self.open_trade
        direction = trade["direction"]
        entry = trade["entry_price"]

        if direction == "LONG":
            trade["max_favorable_price"] = max(trade["max_favorable_price"], current_price)
            unrealized_pct = (current_price - entry) / entry * 100
        else:
            trade["max_favorable_price"] = min(trade["max_favorable_price"], current_price)
            unrealized_pct = (entry - current_price) / entry * 100

        if unrealized_pct >= config.TRAIL_ACTIVATE_PCT and not trade["trail_active"]:
            trade["trail_active"] = True
            print(f"  [TRAIL] Activated at {unrealized_pct:.2f}% profit", flush=True)

        if trade["trail_active"]:
            if direction == "LONG":
                new_sl = trade["max_favorable_price"] * (1 - config.TRAIL_DISTANCE_PCT / 100)
                old_sl = trade.get("trail_sl_price") or trade["sl_price"]
                if new_sl > old_sl:
                    trade["trail_sl_price"] = new_sl
            else:
                new_sl = trade["max_favorable_price"] * (1 + config.TRAIL_DISTANCE_PCT / 100)
                old_sl = trade.get("trail_sl_price") or trade["sl_price"]
                if new_sl < old_sl:
                    trade["trail_sl_price"] = new_sl

    def get_effective_sl(self) -> float | None:
        if not self.open_trade:
            return None
        return self.open_trade.get("trail_sl_price") or self.open_trade["sl_price"]

    # ── Daily reset ──────────────────────────────────────────────

    def check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self.daily_date:
            self._log_daily()
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_date = today

    def _log_daily(self):
        print(f"\n  [DAILY] {self.daily_date} | "
              f"Trades: {self.daily_trades} | "
              f"PnL: ${self.daily_pnl:.2f} | "
              f"Equity: ${self.current_equity:.2f}", flush=True)

    # ── Statistics ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        n = len(self.trades)
        if n == 0:
            return {
                "total_trades": 0, "wr_all": 0.0, "wr_last20": 0.0,
                "total_pnl": 0.0, "equity": self.current_equity,
                "peak_equity": self.peak_equity, "max_drawdown": self.max_drawdown,
                "current_dd": 0.0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
            }

        wins = [t for t in self.trades if t.get("pnl_usd", 0) > 0]
        losses = [t for t in self.trades if t.get("pnl_usd", 0) <= 0]

        last20 = self.trades[-20:] if n >= 20 else self.trades
        wins_20 = sum(1 for t in last20 if t.get("pnl_usd", 0) > 0)

        total_pnl = sum(t.get("pnl_usd", 0) for t in self.trades)
        gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        current_dd = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0

        return {
            "total_trades": n,
            "wr_all": len(wins) / n,
            "wr_last20": wins_20 / len(last20) if last20 else 0,
            "total_pnl": total_pnl,
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "current_dd": current_dd,
            "profit_factor": pf,
            "avg_win": gross_profit / len(wins) if wins else 0,
            "avg_loss": -gross_loss / len(losses) if losses else 0,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
        }

    def log_status(self):
        s = self.get_stats()
        parts = [
            f"Trades={s['total_trades']}",
            f"WR={s['wr_all']:.0%}",
            f"WR20={s['wr_last20']:.0%}",
            f"PnL=${s['total_pnl']:.2f}",
            f"Eq=${s['equity']:.2f}",
            f"MaxDD={s['max_drawdown']:.1%}",
            f"DD={s['current_dd']:.1%}",
            f"PF={s['profit_factor']:.2f}",
        ]
        print(f"  [STATS] {' | '.join(parts)}", flush=True)

    def log_report(self):
        s = self.get_stats()
        print(f"\n{'='*65}", flush=True)
        print(f"PERFORMANCE REPORT — {_ts()}", flush=True)
        print(f"{'='*65}", flush=True)
        print(f"  Equity:      ${s['equity']:.2f} (peak ${s['peak_equity']:.2f})", flush=True)
        print(f"  Total PnL:   ${s['total_pnl']:.2f}", flush=True)
        print(f"  Trades:      {s['total_trades']}", flush=True)
        print(f"  Win rate:    {s['wr_all']:.1%} (last 20: {s['wr_last20']:.1%})", flush=True)
        print(f"  Max DD:      {s['max_drawdown']:.1%}", flush=True)
        print(f"  PF:          {s['profit_factor']:.2f}", flush=True)
        print(f"  Avg win:     ${s['avg_win']:.2f}", flush=True)
        print(f"  Avg loss:    ${s['avg_loss']:.2f}", flush=True)
        print(f"  Signals:     {self.signals_generated} gen / "
              f"{self.signals_executed} exec / {self.signals_filtered} filtered", flush=True)
        print(f"{'='*65}\n", flush=True)
