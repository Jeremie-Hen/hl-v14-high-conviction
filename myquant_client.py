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
    predicted_price = lp.get("predictedPrice") or lp.get("predicted_price")
    is_correct = lp.get("isCorrect")
    reasoning = lp.get("reasoning", "")
    reasoning_hash = hashlib.md5(reasoning.encode()).hexdigest()[:8] if reasoning else "none"
    fingerprint = f"{raw_dir}|{confidence}|{predicted_price}|{reasoning_hash}"

    stats = raw.get("stats") or {}
    api_accuracy = stats.get("accuracy")
    total_preds = stats.get("totalPredictions", 0)

    return {
        "name": name.strip(),
        "direction": direction,
        "raw_direction": raw_dir,
        "confidence": confidence,
        "predicted_price": predicted_price,
        "is_correct": is_correct,
        "fingerprint": fingerprint,
        "reasoning_snippet": reasoning[:80] if reasoning else "",
        "api_accuracy": api_accuracy / 100.0 if api_accuracy is not None else None,
        "total_predictions": total_preds,
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


def _pick_sentinel_agents(agents: list[dict], n: int = 3) -> dict[str, dict]:
    """Pick N agents with the longest reasoning as sentinels for content verification."""
    ranked = sorted(agents, key=lambda a: len(a.get("reasoning_snippet", "")), reverse=True)
    return {a["name"]: a for a in ranked[:n]}


def _verify_content_changed(baseline_sentinels: dict, new_agents: list[dict]) -> tuple[int, int]:
    """Compare sentinel agents' content (reasoning + price) to verify genuine change.
    Returns (changed_count, checked_count)."""
    new_by_name = {a["name"]: a for a in new_agents}
    changed = 0
    checked = 0
    for name, old in baseline_sentinels.items():
        new = new_by_name.get(name)
        if not new:
            continue
        checked += 1
        old_fp = old["fingerprint"]
        new_fp = new["fingerprint"]
        if old_fp != new_fp:
            changed += 1
            log(f"  [VERIFY] {name}: CHANGED "
                f"(dir {old['raw_direction']}->{new['raw_direction']}, "
                f"price {old.get('predicted_price')}->{new.get('predicted_price')}, "
                f"reasoning {'changed' if old.get('reasoning_snippet','')[:40] != new.get('reasoning_snippet','')[:40] else 'same prefix'})")
        else:
            log(f"  [VERIFY] {name}: unchanged")
    return changed, checked


def poll_for_fresh_predictions(last_fingerprint: str | None = None) -> tuple[list[dict] | None, dict]:
    """
    Poll myquant.gg until predictions genuinely change for the new hour.

    Key design: ALWAYS captures a fresh baseline at the start of each poll cycle
    (what the API looks like RIGHT NOW before agents regenerate). Then waits for
    the combined fingerprint to differ from this baseline. This prevents false
    positives from cross-hour drift in the stored last_fingerprint.

    Agents typically generate ~71s past the hour.

    Returns (parsed_agents, poll_stats) or (None, poll_stats) on timeout.
    """
    stats = {"polls": 0, "errors": 0, "stale": 0, "generating": 0,
             "elapsed_sec": 0, "sentinel_changed": 0, "sentinel_checked": 0}
    start = time.time()
    deadline = start + config.POLL_WINDOW_SEC

    # ALWAYS capture a fresh baseline — this is what the API looks like RIGHT NOW
    # (pre-generation state). We'll wait for it to change from THIS, not from
    # the stored last_fingerprint which may have drifted.
    baseline_fp = None
    baseline_sentinels = {}
    log("  [POLL] Capturing pre-generation baseline...")
    for attempt in range(3):
        raw = fetch_predictions()
        if raw and len(raw) > 0:
            baseline_agents = parse_all_agents(raw)
            if baseline_agents:
                baseline_fp = build_fingerprint(baseline_agents)
                baseline_sentinels = _pick_sentinel_agents(baseline_agents)
                sentinel_names = list(baseline_sentinels.keys())
                log(f"  [POLL] Baseline: {len(baseline_agents)} agents, fp={baseline_fp[:12]}...")
                log(f"  [POLL] Sentinel agents: {sentinel_names}")
                for sn, sv in baseline_sentinels.items():
                    log(f"    {sn}: dir={sv['raw_direction']} price={sv.get('predicted_price')} "
                        f"reason={sv.get('reasoning_snippet', '')[:50]}...")
                break
        time.sleep(2)

    if baseline_fp is None:
        log("  [POLL] WARNING: Could not capture baseline, falling back to stored fingerprint")
        baseline_fp = last_fingerprint

    while time.time() < deadline:
        stats["polls"] += 1
        raw = fetch_predictions()

        if raw is None:
            stats["errors"] += 1
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        if len(raw) == 0:
            stats["errors"] += 1
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        agents = parse_all_agents(raw)
        ready_count = len(agents)
        not_ready_count = len(raw) - ready_count

        if ready_count < config.MIN_AGENTS_READY:
            stats["generating"] += 1
            if stats["generating"] == 1:
                deadline = max(deadline, time.time() + 120)
                log(f"  [POLL] Waiting for agents: {ready_count}/{len(raw)} ready "
                    f"(need >={config.MIN_AGENTS_READY}) — extending deadline")
            elif stats["generating"] % 12 == 0:
                remaining = deadline - time.time()
                log(f"  [POLL] Still waiting: {ready_count}/{len(raw)} ready "
                    f"({stats['generating']} checks, {remaining:.0f}s remaining)")
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        if not agents:
            stats["errors"] += 1
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        fp = build_fingerprint(agents)
        if baseline_fp and fp == baseline_fp:
            stats["stale"] += 1
            if stats["stale"] <= 3 or stats["stale"] % 20 == 0:
                log(f"  [POLL] Stale (attempt {stats['polls']}, {stats['stale']} stale)")
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        # Fingerprint changed — verify with sentinel content comparison
        changed, checked = _verify_content_changed(baseline_sentinels, agents)
        stats["sentinel_changed"] = changed
        stats["sentinel_checked"] = checked

        if checked > 0 and changed == 0:
            stats["stale"] += 1
            log(f"  [POLL] Fingerprint changed but sentinel content unchanged "
                f"(0/{checked} sentinels) — treating as stale, attempt {stats['polls']}")
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        elapsed = time.time() - start
        stats["elapsed_sec"] = round(elapsed, 1)
        log(f"  [POLL] Fresh predictions: {len(agents)}/{len(raw)} agents ready in {elapsed:.1f}s "
            f"({stats['polls']} polls, {stats['errors']} errors, "
            f"{stats['generating']} generating, "
            f"{changed}/{checked} sentinels verified)")
        return agents, stats

    stats["elapsed_sec"] = round(time.time() - start, 1)
    log(f"  [POLL] Timeout after {stats['elapsed_sec']}s ({stats['polls']} polls, "
        f"{stats['generating']} generating, {stats['stale']} stale)")
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
          agent_{name}_acc_72h: rolling accuracy (local history, or API stats fallback)
        """
        features = {}
        local_acc_count = 0
        api_acc_count = 0

        import re
        for agent in agents:
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', agent["name"])
            safe_name = re.sub(r'_+', '_', safe_name).strip('_')
            features[f"agent_{safe_name}_dir"] = agent["direction"]

            # Use local rolling accuracy if available (more recent / granular)
            acc = self.get_rolling_accuracy(agent["name"])
            if acc is not None:
                features[f"agent_{safe_name}_acc_{config.ACC_WINDOW}h"] = acc
                local_acc_count += 1
            elif agent.get("api_accuracy") is not None and agent.get("total_predictions", 0) >= 20:
                features[f"agent_{safe_name}_acc_{config.ACC_WINDOW}h"] = agent["api_accuracy"]
                api_acc_count += 1

        if api_acc_count > 0:
            log(f"  [HIST] Accuracy sources: {local_acc_count} local, {api_acc_count} from API stats")

        return features
