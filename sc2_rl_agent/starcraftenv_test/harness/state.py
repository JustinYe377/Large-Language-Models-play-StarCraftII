"""
Typed game state for the harness, parsed from the bot's raw observation.

raw_observation sections are Python-repr strings ("{'mineral': 65, ...}").
GameState normalises the bot's inconsistent key names (statgate_count,
Zealot_count, ...) into lower-case names and splits research progress into
done / in progress.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _section(raw, key):
    val = raw.get(key) if isinstance(raw, dict) else None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _count_key(key, prefix=""):
    """'planning_Zealot_count' -> 'zealot'; 'statgate_count' -> 'stargate'."""
    name = key.lower()
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith("_count"):
        name = name[:-len("_count")]
    return {"statgate": "stargate", "gas_buildings": "assimilator"}.get(name, name)


def _counts(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[_count_key(k, prefix)] = int(v)
    return out


@dataclass
class GameState:
    game_time: str = "00:00"
    minerals: int = 0
    gas: int = 0
    supply_used: int = 0
    supply_cap: int = 0
    supply_left: int = 0
    workers: int = 0
    army_supply: int = 0
    buildings: Dict[str, int] = field(default_factory=dict)          # finished
    buildings_pending: Dict[str, int] = field(default_factory=dict)  # under construction
    units: Dict[str, int] = field(default_factory=dict)
    units_pending: Dict[str, int] = field(default_factory=dict)
    research_done: List[str] = field(default_factory=list)
    research_in_progress: Dict[str, float] = field(default_factory=dict)
    enemy_units: Dict[str, int] = field(default_factory=dict)
    enemy_structures: Dict[str, int] = field(default_factory=dict)
    # From the env observation (bot v2), None when unknown.
    scouts_out: Optional[int] = None
    chrono_ready: Optional[bool] = None

    def has(self, building, n=1):
        return self.buildings.get(building, 0) >= n

    def has_or_pending(self, building):
        return self.buildings.get(building, 0) + self.buildings_pending.get(building, 0) > 0

    def research_state(self, key):
        """'done' | 'in_progress' | None"""
        if key in self.research_done:
            return "done"
        if key in self.research_in_progress:
            return "in_progress"
        return None


def parse_state(raw, observation=None) -> Optional[GameState]:
    """`observation` is the full env observation; bot-v2 extras (scouts_out,
    chrono_ready) are read from it when present."""
    if not isinstance(raw, dict):
        return None
    extra = observation if isinstance(observation, dict) else {}
    res = _section(raw, "resource")
    planning = _section(raw, "planning")
    research = _section(raw, "research")
    enemy = _section(raw, "enemy")

    done, in_progress = [], {}
    for building, items in research.items():
        if not isinstance(items, dict):
            continue
        for key, v in items.items():
            if not isinstance(v, (int, float)):
                continue
            name = key.lower().replace("_research_status", "")
            if v >= 1:
                done.append(name)
            elif v > 0:
                in_progress[name] = round(float(v), 2)

    def _enemy(section):
        d = enemy.get(section, {}) if isinstance(enemy, dict) else {}
        return {str(k).lower(): int(v) for k, v in d.items() if isinstance(v, (int, float))} if isinstance(d, dict) else {}

    return GameState(
        game_time=str(res.get("game_time", "00:00")),
        minerals=int(res.get("mineral", 0) or 0),
        gas=int(res.get("gas", 0) or 0),
        supply_used=int(res.get("supply_used", 0) or 0),
        supply_cap=int(res.get("supply_cap", 0) or 0),
        supply_left=int(res.get("supply_left", 0) or 0),
        workers=int(res.get("worker_supply", 0) or 0),
        army_supply=int(res.get("army_supply", 0) or 0),
        buildings=_counts(_section(raw, "building")),
        buildings_pending=_counts(planning.get("planning_structure", {}) if isinstance(planning, dict) else {},
                                  "planning_"),
        units=_counts(_section(raw, "unit")),
        units_pending=_counts(planning.get("planning_unit", {}) if isinstance(planning, dict) else {},
                              "planning_"),
        research_done=sorted(done),
        research_in_progress=in_progress,
        enemy_units=_enemy("unit"),
        enemy_structures=_enemy("structure"),
        scouts_out=extra.get("scouts_out"),
        chrono_ready=extra.get("chrono_ready"),
    )


def _fmt_counts(d):
    items = [f"{k} {v}" for k, v in sorted(d.items()) if v]
    return ", ".join(items) if items else "none"


def render_state(s: GameState) -> str:
    """Compact, stable-order snapshot for the prompt."""
    lines = [
        f"time {s.game_time} | minerals {s.minerals} | gas {s.gas} | "
        f"supply {s.supply_used}/{s.supply_cap} (free {s.supply_left}) | workers {s.workers} | army supply {s.army_supply}",
        f"buildings: {_fmt_counts(s.buildings)}",
        f"under construction: {_fmt_counts(s.buildings_pending)}",
        f"units: {_fmt_counts({k: v for k, v in s.units.items() if k != 'probe'})}",
        f"units in production: {_fmt_counts(s.units_pending)}",
        "research done: " + (", ".join(s.research_done) or "none"),
        "research in progress: " + (", ".join(f"{k} {int(v * 100)}%" for k, v in sorted(s.research_in_progress.items()))
                                    or "none"),
        f"enemy seen: units {_fmt_counts(s.enemy_units)}; structures {_fmt_counts(s.enemy_structures)}",
    ]
    return "\n".join(lines)


def diff_states(prev: Optional[GameState], cur: GameState) -> List[str]:
    """Human-readable changes since the previous decision (empty for the first)."""
    if prev is None:
        return []
    out = []
    for label, a, b in (("buildings", prev.buildings, cur.buildings), ("units", prev.units, cur.units),
                        ("enemy units seen", prev.enemy_units, cur.enemy_units),
                        ("enemy structures seen", prev.enemy_structures, cur.enemy_structures)):
        changes = []
        for k in sorted(set(a) | set(b)):
            d = b.get(k, 0) - a.get(k, 0)
            if d:
                changes.append(f"{k} {d:+d}")
        if changes:
            out.append(f"{label}: " + ", ".join(changes))
    finished = sorted(set(cur.research_done) - set(prev.research_done))
    if finished:
        out.append("research finished: " + ", ".join(finished))
    started = sorted(set(cur.research_in_progress) - set(prev.research_in_progress) - set(cur.research_done))
    if started:
        out.append("research started: " + ", ".join(started))
    if cur.supply_left == 0 and cur.supply_cap < 200:
        out.append("SUPPLY BLOCKED")
    return out
