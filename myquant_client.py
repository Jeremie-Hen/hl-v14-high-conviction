"""
myquant.gg API Client — v14 High-Conviction Bot
Fetches all agent predictions, polls for fresh data, and builds feature matrices.
"""
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

AGENT_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "agent_history.json")

_DIRECTION_MAP = {
    "LONG": 1, "SHORT": -1,
    "UP": 1, "DOWN": -1,
    "BUY": 1, "SELL": -1,
    "BULLISH": 1, "BEARISH": -1,
}


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def fetch_predictions(timeout: int = 15) -> list | None:
    """Fetch latest predictions from myquant.gg. Returns list or None on error."""
    try:
        url = config.MYQUANT_API_URL
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}_t={int(time.time())}"
        resp = requests.get(url, timeout=timeout, headers={"Cache-Control": "no-cache"})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("predictions", data.get("data", []))
        return []
    except Exception as e:
        log(f"  [API] Error fetching predictions: {e}")
        return None


def _parse_agent(raw: dict) -> dict | None:
    """Parse one agent record into standardized format."""
    name = raw.get("agentName") or raw.get("agent_name") or raw.get("name") or ""
    if not name:
        return None

    lp = raw.get("latestPrediction") or {}
    raw_dir = str(lp.get("direction") or raw.get("direction") or raw.get("prediction") or "").upper().strip()
    direction = _DIRECTION_MAP.get(raw_dir, 0)

    confidence = lp.get("confidence") or raw.get("confidence") or 0
    is_correct = lp.get("isCorrect")
    reasoning = lp.get("reasoning", "")
    reasoning_hash = hashlib.md5(reasoning.encode()).hexdigest()[:8] if reasoning else "none"
    fingerprint = f"{raw_dir}|{confidence}|{reasoning_hash}"

    return {
        "name": name.strip(),
        "direction": direction,
        "raw_direction": raw_dir,
        "confidence": confidence,
        "is_correct": is_correct,
        "fingerprint": fingerprint,
    }


def parse_all_agents(predictions: list) -> list[dict]:
    """Parse all agent predictions into a list of dicts."""
    agents = []
    for raw in predictions:
        parsed = _parse_agent(raw)
        if parsed and parsed["direction"] != 0:
            agents.append(parsed)
    return agents


def build_fingerprint(agents: list[dict]) -> str:
    """Build a combined fingerprint for all agents to detect staleness."""
    fps = sorted(f"{a['name']}:{a['fingerprint']}" for a in agents)
    return hashlib.md5("|".join(fps).encode()).hexdigest()


def poll_for_fresh_predictions(last_fingerprint: str | None = None) -> tuple[list[dict] | None, dict]:
    """
    Poll myquant.gg until predictions change (new fingerprint).
    Returns (parsed_agents, poll_stats) or (None, poll_stats) on timeout.
    """
    stats = {"polls": 0, "errors": 0, "stale": 0, "elapsed_sec": 0}
    start = time.time()
    deadline = start + config.POLL_WINDOW_SEC

    while time.time() < deadline:
        stats["polls"] += 1
        raw = fetch_predictions()

        if raw is None:
            stats["errors"] += 1
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        agents = parse_all_agents(raw)
        if not agents:
            stats["errors"] += 1
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        fp = build_fingerprint(agents)
        if last_fingerprint and fp == last_fingerprint:
            stats["stale"] += 1
            if stats["stale"] <= 3 or stats["stale"] % 20 == 0:
                log(f"  [POLL] Stale (attempt {stats['polls']}, {stats['stale']} stale)")
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        elapsed = time.time() - start
        stats["elapsed_sec"] = round(elapsed, 1)
        log(f"  [POLL] Fresh predictions: {len(agents)} agents in {elapsed:.1f}s "
            f"({stats['polls']} polls, {stats['errors']} errors)")
        return agents, stats

    stats["elapsed_sec"] = round(time.time() - start, 1)
    log(f"  [POLL] Timeout after {stats['elapsed_sec']}s ({stats['polls']} polls)")
    return None, stats


# ═════════════════════════════════════════════════════════════════════
# AGENT HISTORY & ROLLING ACCURACY
# ═════════════════════════════════════════════════════════════════════

class AgentHistoryTracker:
    """Track per-agent prediction history and compute rolling accuracy."""

    def __init__(self):
        self.history: dict[str, list[dict]] = defaultdict(list)
        self._load()

    def _load(self):
        if os.path.exists(AGENT_HISTORY_PATH):
            try:
                with open(AGENT_HISTORY_PATH, "r") as f:
                    data = json.load(f)
                self.history = defaultdict(list, data)
                total = sum(len(v) for v in self.history.values())
                log(f"  [HIST] Loaded {total} records for {len(self.history)} agents")
            except Exception as e:
                log(f"  [HIST] Error loading: {e}")

    def save(self):
        try:
            trimmed = {}
            for name, records in self.history.items():
                trimmed[name] = records[-(config.ACC_WINDOW * 3):]
            with open(AGENT_HISTORY_PATH, "w") as f:
                json.dump(trimmed, f)
        except Exception as e:
            log(f"  [HIST] Error saving: {e}")

    def record_predictions(self, agents: list[dict], hour_ts: str):
        """Store current hour's predictions."""
        for agent in agents:
            self.history[agent["name"]].append({
                "ts": hour_ts,
                "dir": agent["direction"],
                "is_correct": agent["is_correct"],
            })

    def get_rolling_accuracy(self, agent_name: str, window: int = None) -> float | None:
        """Compute rolling accuracy from isCorrect field over last `window` records."""
        window = window or config.ACC_WINDOW
        records = self.history.get(agent_name, [])
        if not records:
            return None

        recent = records[-window:]
        evaluated = [r for r in recent if r.get("is_correct") is not None]
        if len(evaluated) < 10:
            return None

        return sum(1 for r in evaluated if r["is_correct"]) / len(evaluated)

    def build_agent_features(self, agents: list[dict]) -> dict:
        """
        Build feature dict for the current hour:
          agent_{name}_dir: +1/-1
          agent_{name}_acc_72h: rolling accuracy
        """
        features = {}
        for agent in agents:
            safe_name = agent["name"].replace(" ", "_").replace("-", "_")
            features[f"agent_{safe_name}_dir"] = agent["direction"]
            acc = self.get_rolling_accuracy(agent["name"])
            if acc is not None:
                features[f"agent_{safe_name}_acc_{config.ACC_WINDOW}h"] = acc
        return features
