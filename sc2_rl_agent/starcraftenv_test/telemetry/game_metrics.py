"""
Macro metrics pulled out of a raw observation.

raw_observation sections ('resource', 'building', 'unit', ...) are Python
repr strings like "{'mineral': 65, 'gas': 4, ...}", so they go through
ast.literal_eval rather than json.
"""

import ast
import math
from typing import Dict, Optional

from .schema import GAME_LOOPS_PER_SECOND

WORKER_UNITS = {"probe", "drone", "scv"}
BASE_KEYS = ("nexus_count", "hatchery_count", "lair_count", "hive_count",
             "base_count", "command_center_count")

# raw 'resource' key -> metric name
_RESOURCE_FIELDS = {
    "mineral": "minerals",
    "gas": "gas",
    "supply_used": "supply_used",
    "supply_cap": "supply_cap",
    "supply_left": "supply_left",
    "worker_supply": "worker_supply",
    "army_supply": "army_supply",
}


def _section(raw_obs, key) -> Dict:
    if not isinstance(raw_obs, dict):
        return {}
    val = raw_obs.get(key)
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    return None


def mmss_to_seconds(text) -> Optional[float]:
    try:
        m, s = str(text).split(":")
        return float(int(m) * 60 + int(s))
    except (ValueError, TypeError):
        return None


def game_time_seconds(raw_obs, game_loop=None) -> Optional[float]:
    """Prefer the exact game loop; fall back to the 'MM:SS' string."""
    if isinstance(game_loop, (int, float)) and not isinstance(game_loop, bool):
        return round(game_loop / GAME_LOOPS_PER_SECOND, 3)
    return mmss_to_seconds(_section(raw_obs, "resource").get("game_time"))


def army_unit_counts(raw_obs) -> Dict[str, int]:
    """Our non-worker unit counts, keyed by lower-cased unit name."""
    counts = {}
    for key, val in _section(raw_obs, "unit").items():
        n = _num(val)
        if n is None or not key.lower().endswith("_count"):
            continue
        name = key.lower()[:-len("_count")]
        if name in WORKER_UNITS:
            continue
        counts[name] = int(n)
    return counts


def shannon_entropy(counts) -> float:
    """Entropy in bits of a count distribution (0 for empty / single type)."""
    total = sum(c for c in counts if c > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def extract_metrics(raw_obs) -> Dict[str, float]:
    """Flat numeric metrics without the `game/` prefix."""
    out: Dict[str, float] = {}
    resource = _section(raw_obs, "resource")
    for raw_key, name in _RESOURCE_FIELDS.items():
        n = _num(resource.get(raw_key))
        if n is not None:
            out[name] = n

    building = _section(raw_obs, "building")
    for key, val in building.items():
        n = _num(val)
        if n is not None:
            out[key] = n
    bases = [building[k] for k in BASE_KEYS if _num(building.get(k)) is not None]
    if bases:
        out["base_count"] = sum(bases)

    army = army_unit_counts(raw_obs)
    out["army_unit_count"] = sum(army.values())
    out["army_comp_entropy"] = shannon_entropy(list(army.values()))

    if "minerals" in out or "gas" in out:
        out["bank"] = out.get("minerals", 0) + out.get("gas", 0)
    if "supply_left" in out and "supply_cap" in out:
        out["supply_blocked"] = int(out["supply_left"] == 0 and out["supply_cap"] < 200)
    return out


class GameMetricsTracker:
    """Keeps the whole-game aggregates needed for the summary."""

    def __init__(self):
        self.peak_supply = 0
        self.peak_workers = 0
        self.max_bases = 0
        self.supply_blocked_s = 0.0
        self._entropy_sum = 0.0
        self._entropy_n = 0
        self._last_time = None
        self._last_blocked = False
        self.last_game_time_s = 0.0

    def update(self, game_time_s, metrics):
        self.peak_supply = max(self.peak_supply, metrics.get("supply_used", 0))
        self.peak_workers = max(self.peak_workers, metrics.get("worker_supply", 0))
        self.max_bases = max(self.max_bases, metrics.get("base_count", 0))
        if metrics.get("army_unit_count", 0) > 0:
            self._entropy_sum += metrics.get("army_comp_entropy", 0.0)
            self._entropy_n += 1
        if game_time_s is not None:
            if self._last_time is not None and self._last_blocked and game_time_s > self._last_time:
                self.supply_blocked_s += game_time_s - self._last_time
            self._last_time = game_time_s
            self.last_game_time_s = max(self.last_game_time_s, game_time_s)
        self._last_blocked = bool(metrics.get("supply_blocked", 0))

    def summary(self):
        return {
            "peak_supply": self.peak_supply,
            "peak_workers": self.peak_workers,
            "max_bases": self.max_bases,
            "supply_blocked_s": round(self.supply_blocked_s, 3),
            # Mean over steps where we owned at least one army unit.
            "army_comp_entropy_mean": (self._entropy_sum / self._entropy_n) if self._entropy_n else 0.0,
        }
