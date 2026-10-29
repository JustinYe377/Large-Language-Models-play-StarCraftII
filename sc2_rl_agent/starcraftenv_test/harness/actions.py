"""
Protoss action specs: prerequisites, costs and a legality check.

legal_actions(state) is the action mask: an action is offered to the model
only if its prerequisites exist right now and the bot would actually act on
it (capacity.py: busy buildings, the bot's pylon/limit rules, ...).
Resource affordability is NOT masked here (the resource filter still handles
it), because the queue executes over the next steps while income keeps
arriving.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from sc2_rl_agent.starcraftenv_test.utils.action_info import ActionDescriptions
from sc2_rl_agent.starcraftenv_test.utils.resource_filter import PROTOSS_ACTION_COSTS

from .capacity import capacity_reason
from .state import GameState

GATEWAY_LIKE = ("gateway", "warp_gate")


@dataclass
class ActionSpec:
    id: int
    name: str
    requires: Tuple[str, ...] = ()            # buildings that must be finished
    requires_any: Tuple[str, ...] = ()        # at least one of these buildings
    research_key: Optional[str] = None        # blocked once done / in progress
    after_research: Optional[str] = None      # previous upgrade level must be done
    check: Optional[Callable[[GameState], Optional[str]]] = None   # extra rule -> reason or None
    note: str = ""

    def cost_text(self):
        c = PROTOSS_ACTION_COSTS.get(self.name)
        if not c:
            return ""
        parts = [f"{c['minerals']}m"]
        if c["gas"]:
            parts.append(f"{c['gas']}g")
        if c["supply"]:
            parts.append(f"{c['supply']} supply")
        return " ".join(parts)


def _need_units(**mins):
    def check(s):
        for unit, n in mins.items():
            if s.units.get(unit, 0) < n:
                return f"needs {n} {unit}"
        return None
    return check


def _need_army(s):
    return None if s.army_supply > 0 else "no army"


def _templar_pair(s):
    n = s.units.get("high_templar", 0) + s.units.get("dark_templar", 0)
    return None if n >= 2 else "needs 2 high/dark templar"


def _gas_slot(s):
    nexus = s.buildings.get("nexus", 0)
    taken = s.buildings.get("assimilator", 0) + s.buildings_pending.get("assimilator", 0)
    return None if taken < 2 * max(nexus, 1) else "all geysers taken"


def _upgrade_line(base_id, name_fmt, key_fmt, building, level2_building):
    specs = []
    for lvl in (1, 2, 3):
        specs.append(ActionSpec(
            base_id + lvl - 1, name_fmt.format(lvl),
            requires=(building,) + ((level2_building,) if lvl > 1 else ()),
            research_key=key_fmt.format(lvl),
            after_research=key_fmt.format(lvl - 1) if lvl > 1 else None))
    return specs


def _build_specs() -> Dict[int, ActionSpec]:
    S = ActionSpec
    specs = [
        # --- units ---
        S(0, "TRAIN PROBE", requires=("nexus",)),
        S(1, "TRAIN ZEALOT", requires_any=GATEWAY_LIKE),
        S(2, "TRAIN ADEPT", requires=("cybernetics_core",), requires_any=GATEWAY_LIKE),
        S(3, "TRAIN STALKER", requires=("cybernetics_core",), requires_any=GATEWAY_LIKE),
        S(4, "TRAIN SENTRY", requires=("cybernetics_core",), requires_any=GATEWAY_LIKE),
        S(5, "TRAIN HIGHTEMPLAR", requires=("templar_archives",), requires_any=GATEWAY_LIKE),
        S(6, "TRAIN DARKTEMPLAR", requires=("dark_shrine",), requires_any=GATEWAY_LIKE),
        S(7, "TRAIN VOIDRAY", requires=("stargate",)),
        S(8, "TRAIN CARRIER", requires=("stargate", "fleet_beacon")),
        S(9, "TRAIN TEMPEST", requires=("stargate", "fleet_beacon")),
        S(10, "TRAIN ORACLE", requires=("stargate",)),
        S(11, "TRAIN PHOENIX", requires=("stargate",)),
        S(12, "TRAIN MOTHERSHIP", requires=("nexus", "fleet_beacon"),
          check=lambda s: "only one mothership" if s.units.get("mothership", 0) else None),
        S(13, "TRAIN OBSERVER", requires=("robotics_facility",)),
        S(14, "TRAIN IMMORTAL", requires=("robotics_facility",)),
        S(15, "TRAIN WARPPRISM", requires=("robotics_facility",)),
        S(16, "TRAIN COLOSSUS", requires=("robotics_facility", "robotics_bay")),
        S(17, "TRAIN DISRUPTOR", requires=("robotics_facility", "robotics_bay")),
        S(18, "MORPH ARCHON", check=_templar_pair),
        # --- structures (everything but pylon/assimilator/nexus needs pylon power) ---
        S(19, "BUILD PYLON"),
        S(20, "BUILD ASSIMILATOR", requires=("nexus",), check=_gas_slot),
        S(21, "BUILD NEXUS"),
        S(22, "BUILD GATEWAY", requires=("pylon",)),
        S(23, "BUILD CYBERNETICSCORE", requires=("pylon",), requires_any=GATEWAY_LIKE),
        S(24, "BUILD FORGE", requires=("pylon",)),
        S(25, "BUILD TWILIGHTCOUNCIL", requires=("pylon", "cybernetics_core")),
        S(26, "BUILD ROBOTICSFACILITY", requires=("pylon", "cybernetics_core")),
        S(27, "BUILD STARGATE", requires=("pylon", "cybernetics_core")),
        S(28, "BUILD TEMPLARARCHIVE", requires=("pylon", "twilight_council")),
        S(29, "BUILD DARKSHRINE", requires=("pylon", "twilight_council")),
        S(30, "BUILD ROBOTICSBAY", requires=("pylon", "robotics_facility")),
        S(31, "BUILD FLEETBEACON", requires=("pylon", "stargate")),
        # SC2 prerequisites; legacy bot rules add the other building (capacity.py).
        S(32, "BUILD PHOTONCANNON", requires=("pylon", "forge")),
        S(33, "BUILD SHIELDBATTERY", requires=("pylon", "cybernetics_core")),
        # --- research ---
        S(34, "RESEARCH WARPGATERESEARCH", requires=("cybernetics_core",), research_key="warpgate"),
        *_upgrade_line(35, "RESEARCH PROTOSSAIRWEAPONSLEVEL{}", "protoss_air_weapon_level_{}",
                       "cybernetics_core", "fleet_beacon"),
        *_upgrade_line(38, "RESEARCH PROTOSSAIRARMORSLEVEL{}", "protoss_air_armor_level_{}",
                       "cybernetics_core", "fleet_beacon"),
        S(41, "RESEARCH ADEPTPIERCINGATTACK", requires=("twilight_council",), research_key="resonating_glaives"),
        S(42, "RESEARCH BLINKTECH", requires=("twilight_council",), research_key="blink"),
        S(43, "RESEARCH CHARGE", requires=("twilight_council",), research_key="charge"),
        *_upgrade_line(44, "RESEARCH PROTOSSGROUNDWEAPONSLEVEL{}", "protoss_ground_weapon_level_{}",
                       "forge", "twilight_council"),
        *_upgrade_line(47, "RESEARCH PROTOSSGROUNDARMORSLEVEL{}", "protoss_ground_armor_level_{}",
                       "forge", "twilight_council"),
        *_upgrade_line(50, "RESEARCH PROTOSSSHIELDSLEVEL{}", "protoss_shield_level_{}",
                       "forge", "twilight_council"),
        S(53, "RESEARCH EXTENDEDTHERMALLANCE", requires=("robotics_bay",), research_key="extended_thermal_lance"),
        S(54, "RESEARCH GRAVITICDRIVE", requires=("robotics_bay",), research_key="graviticdrive"),
        S(55, "RESEARCH OBSERVERGRAVITICBOOSTER", requires=("robotics_bay",), research_key="observergraviticbooster"),
        S(56, "RESEARCH PSISTORMTECH", requires=("templar_archives",), research_key="psistormtech"),
        S(57, "RESEARCH VOIDRAYSPEEDUPGRADE", requires=("fleet_beacon",), research_key="voidrayspeedupgrade"),
        S(58, "RESEARCH PHOENIXRANGEUPGRADE", requires=("fleet_beacon",), research_key="phoenixrangeupgrade"),
        S(59, "RESEARCH TEMPESTGROUNDATTACKUPGRADE", requires=("fleet_beacon",)),
        # --- other ---
        S(60, "SCOUTING PROBE", check=_need_units(probe=1)),
        S(61, "SCOUTING OBSERVER", check=_need_units(observer=1)),
        S(62, "SCOUTING ZEALOT", check=_need_units(zealot=1)),
        S(63, "SCOUTING PHOENIX", check=_need_units(phoenix=1)),
        S(64, "MULTI-ATTACK", check=_need_army, note="send the whole army to attack"),
        S(65, "MULTI-RETREAT", check=_need_army, note="pull the army back home"),
        S(66, "CHRONOBOOST NEXUS", requires=("nexus",), note="speeds up the nexus while it trains a probe"),
        S(67, "CHRONOBOOST CYBERNETICSCORE", requires=("nexus", "cybernetics_core"), note="while it researches"),
        S(68, "CHRONOBOOST TWILIGHTCOUNCIL", requires=("nexus", "twilight_council"), note="while it researches"),
        S(69, "CHRONOBOOST STARGATE", requires=("nexus", "stargate"), note="while it trains"),
        S(70, "CHRONOBOOST FORGE", requires=("nexus", "forge"), note="while it researches"),
        S(71, "EMPTY ACTION", note="do nothing"),
    ]
    out = {s.id: s for s in specs}
    vocab = ActionDescriptions("Protoss").flattened_actions
    assert {i: s.name for i, s in out.items()} == dict(vocab), "action specs out of sync with action_info"
    return out


PROTOSS_ACTIONS: Dict[int, ActionSpec] = _build_specs()
EMPTY_ACTION_ID = 71


def illegal_reason(spec: ActionSpec, s: GameState, rules: str = "dynamic") -> Optional[str]:
    missing = [b for b in spec.requires if not s.has(b)]
    if missing:
        return "needs " + " and ".join(missing)
    if spec.requires_any and not any(s.has(b) for b in spec.requires_any):
        return "needs " + " or ".join(spec.requires_any)
    if spec.research_key:
        st = s.research_state(spec.research_key)
        if st == "done":
            return "already researched"
        if st == "in_progress":
            return "already researching"
    if spec.after_research and s.research_state(spec.after_research) != "done":
        return f"needs {spec.after_research} first"
    if spec.check:
        reason = spec.check(s)
        if reason:
            return reason
    return capacity_reason(spec.id, spec.name, s, rules)


def legal_actions(s: GameState, rules: str = "dynamic") -> Dict[int, Optional[str]]:
    """id -> None if legal, else the reason it is masked. `rules` is the bot's
    bot_rules mode ("dynamic" or "legacy")."""
    return {i: illegal_reason(spec, s, rules) for i, spec in PROTOSS_ACTIONS.items()}


def catalog_text() -> str:
    """Static action catalog for the system prompt (identical every call)."""
    lines = []
    for spec in PROTOSS_ACTIONS.values():
        req = list(spec.requires)
        if spec.requires_any:
            req.append(" or ".join(spec.requires_any))
        if spec.after_research:
            req.append(spec.after_research)
        bits = [f"{spec.id}: {spec.name}"]
        cost = spec.cost_text()
        if cost:
            bits.append(f"cost {cost}")
        if req:
            bits.append("requires " + ", ".join(req))
        if spec.note:
            bits.append(spec.note)
        lines.append(" | ".join(bits))
    return "\n".join(lines)
