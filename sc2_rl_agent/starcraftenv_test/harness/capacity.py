"""
Capacity / bot-rule masking: hide actions the bot would refuse right now,
even though their prerequisites exist.

Each rule mirrors the corresponding handler in env/bot/Protoss_bot.py (bot
v2), for both `bot_rules` modes, so the mask predicts what the bot will
actually do, not what SC2 would allow.
Things the observation cannot show (Nexus energy, existing chrono buffs,
warp-gate cooldowns, building placement, the scouting interval) are not
masked; those actions can still fail.

Every rule returns a reason string when the action is masked, else None.
"""

from typing import Callable, Dict, Optional

from sc2_rl_agent.starcraftenv_test.utils.resource_filter import PROTOSS_ACTION_COSTS

from .state import GameState

ROBO_UNITS = ("observer", "immortal", "warp_prism", "colossus", "disruptor")
STARGATE_UNITS = ("phoenix", "oracle", "voidray", "carrier", "tempest")

# research key -> building that researches it (one research at a time each)
RESEARCH_BUILDING = {
    "warpgate": "cybernetics_core",
    **{f"protoss_air_{kind}_level_{n}": "cybernetics_core" for kind in ("weapon", "armor") for n in (1, 2, 3)},
    **{f"protoss_ground_{kind}_level_{n}": "forge" for kind in ("weapon", "armor") for n in (1, 2, 3)},
    **{f"protoss_shield_level_{n}": "forge" for n in (1, 2, 3)},
    "blink": "twilight_council", "charge": "twilight_council", "resonating_glaives": "twilight_council",
    "extended_thermal_lance": "robotics_bay", "graviticdrive": "robotics_bay",
    "observergraviticbooster": "robotics_bay",
    "phoenixrangeupgrade": "fleet_beacon", "voidrayspeedupgrade": "fleet_beacon",
    "psistormtech": "templar_archives", "shadowstrike": "dark_shrine",
}

# research action id -> building (ids 35-58; 59 is not implemented by the bot)
RESEARCH_ACTION_BUILDING = {
    34: "cybernetics_core", **{i: "cybernetics_core" for i in range(35, 41)},
    41: "twilight_council", 42: "twilight_council", 43: "twilight_council",
    **{i: "forge" for i in range(44, 53)},
    53: "robotics_bay", 54: "robotics_bay", 55: "robotics_bay",
    56: "templar_archives", 57: "fleet_beacon", 58: "fleet_beacon",
}

# BUILD actions with time-window count limits (handle_action_22..33 building_limits).
# Counts are the bot's: finished + pending of that exact type (warp gates are
# not counted as gateways).
BUILDING_LIMITS = {
    22: ("gateway", {('00:00', '03:00'): 1, ('03:00', '05:00'): 3, ('05:00', '07:00'): 6, ('07:00', '10:00'): 8}),
    23: ("cybernetics_core", {('00:00', '05:00'): 1, ('05:00', '20:00'): 2}),
    24: ("forge", {('00:00', '05:00'): 1, ('05:00', '10:00'): 2}),
    26: ("robotics_facility", {('00:00', '05:00'): 1, ('05:00', '10:00'): 2, ('10:00', '15:00'): 3}),
    27: ("stargate", {('00:00', '05:00'): 1, ('05:00', '10:00'): 3, ('10:00', '15:00'): 4}),
    32: ("photon_cannon", {('00:00', '05:00'): 1, ('05:00', '10:00'): 3, ('10:00', '15:00'): 5, ('15:00', '20:00'): 7}),
    33: ("shield_battery", {('00:00', '05:00'): 1, ('05:00', '10:00'): 3, ('10:00', '15:00'): 5, ('15:00', '20:00'): 7}),
}

# Chronoboost: target building, the bot's supply requirement, and what makes the target busy.
CHRONO = {
    66: ("nexus", 2), 67: ("cybernetics_core", 0), 68: ("twilight_council", 4),
    69: ("stargate", 4), 70: ("forge", 4),
}

UNIT_LIMITS = {13: ("observer", 2), 15: ("warp_prism", 1), 17: ("disruptor", 1),
               10: ("oracle", 1), 11: ("phoenix", 4)}

GATEWAY_UNITS = {1: "zealot", 2: "adept", 3: "stalker", 4: "sentry", 5: "high_templar", 6: "dark_templar"}


def _total(s: GameState, building):
    return s.buildings.get(building, 0) + s.buildings_pending.get(building, 0)


def _pending_units(s: GameState, names):
    return sum(s.units_pending.get(n, 0) for n in names)


def _researching_at(s: GameState, building):
    return sum(1 for k in s.research_in_progress if RESEARCH_BUILDING.get(k) == building)


def _pylon_thresholds(t):
    """handle_action_19: (max free supply, max pending pylons), string-compared like the bot."""
    if '06:00' <= t <= '07:00':
        return 5, 4
    if '06:00' <= t <= '08:00':
        return 5, 3
    if '08:00' <= t <= '10:00':
        return 7, 4
    if '10:00' <= t:
        return 7, 4
    return 3, 2


def _supply_cost(name):
    return PROTOSS_ACTION_COSTS.get(name, {}).get("supply", 0)


def pylon_supply_need(s: GameState):
    """Mirror of Protoss_Bot.pylon_supply_need (dynamic rules)."""
    gates = s.buildings.get("gateway", 0) + s.buildings.get("warp_gate", 0)
    needed = (2 + 2 * s.buildings.get("nexus", 0) + 2 * gates
              + 4 * s.buildings.get("robotics_facility", 0) + 4 * s.buildings.get("stargate", 0))
    covered = (s.supply_left + 8 * s.buildings_pending.get("pylon", 0)
               + 15 * s.buildings_pending.get("nexus", 0))
    return needed, covered


def capacity_reason(action_id: int, name: str, s: GameState, rules: str = "dynamic") -> Optional[str]:
    t = s.game_time
    legacy = rules == "legacy"

    # --- probes (handle_action_0): bot v2 queues one probe behind the current one ---
    if action_id == 0:
        if s.workers + s.units_pending.get("probe", 0) > 75:
            return "probe cap (75) reached"
        if s.supply_left <= 0:
            return "no free supply"
        if s.units_pending.get("probe", 0) >= 2 * s.buildings.get("nexus", 0):
            return "every nexus already has a probe queued"
        return None

    # --- other units need free supply (train_from_* / warp_unit check it) ---
    cost = _supply_cost(name)
    if cost and s.supply_left < cost:
        return f"needs {cost} free supply, have {s.supply_left}"

    # --- gateway units (produce_bg_unit) ---
    if action_id in GATEWAY_UNITS:
        if "warpgate" in s.research_done:
            if not s.buildings.get("warp_gate", 0):
                return "no warp gate yet"
        elif not s.buildings.get("gateway", 0):
            return "needs a gateway (not yet warp gate)"
        return None

    # --- robotics / stargate units: the bot skips them while every building is busy ---
    if action_id in UNIT_LIMITS:
        unit, limit = UNIT_LIMITS[action_id]
        if s.units.get(unit, 0) + s.units_pending.get(unit, 0) >= limit:
            return f"{unit} limit ({limit}) reached"
    if action_id in range(13, 18):
        if _pending_units(s, ROBO_UNITS) >= s.buildings.get("robotics_facility", 0):
            return "every robotics facility is busy"
    if action_id in range(7, 12):
        if _pending_units(s, STARGATE_UNITS) >= s.buildings.get("stargate", 0):
            return "every stargate is busy"

    # --- archon (handle_action_18 morphs high templar if any exist, else dark templar) ---
    if action_id == 18:
        ht, dt = s.units.get("high_templar", 0), s.units.get("dark_templar", 0)
        if not (ht >= 2 or (ht == 0 and dt >= 2)):
            return "needs 2 high templar (or 2 dark templar and no high templar)"

    # --- pylon (handle_action_19) ---
    if action_id == 19:
        if s.supply_cap >= 200:
            return "supply cap is already 200"
        if legacy:
            max_free, max_pending = _pylon_thresholds(t)
            if s.supply_left > max_free:
                return f"not needed yet: the bot builds pylons only when free supply <= {max_free}"
            if s.buildings_pending.get("pylon", 0) > max_pending:
                return f"{s.buildings_pending.get('pylon', 0)} pylons already under construction"
        else:
            needed, covered = pylon_supply_need(s)
            if covered >= needed:
                return (f"supply is sufficient ({covered} free or under construction, "
                        f"{needed} needed for one production cycle)")

    # --- assimilator (handle_action_20) ---
    if action_id == 20 and s.buildings_pending.get("assimilator", 0) >= 2:
        return "2 assimilators already under construction"

    # --- nexus (handle_action_21) ---
    if action_id == 21 and (s.buildings_pending.get("nexus", 0) > 1 or s.buildings.get("nexus", 0) > 8):
        return "a nexus is already under construction"

    # --- legacy only: time-window building limits and the extra cannon/battery prerequisite ---
    if legacy and action_id in BUILDING_LIMITS:
        building, windows = BUILDING_LIMITS[action_id]
        count = _total(s, building) + (s.buildings.get("warp_gate", 0) if building == "gateway" else 0)
        for (start, end), limit in windows.items():
            if start <= t <= end and count >= limit:
                return f"limit of {limit} {building} until {end}"
    if legacy and action_id == 32 and not s.has("cybernetics_core"):
        return "needs cybernetics_core (legacy rules)"
    if legacy and action_id == 33 and not s.has("forge"):
        return "needs forge (legacy rules)"
    if action_id in (32, 33) and _total(s, "photon_cannon" if action_id == 32 else "shield_battery") > 3:
        return "limit of 4 reached"

    # --- research: one at a time per building ---
    if action_id in RESEARCH_ACTION_BUILDING:
        building = RESEARCH_ACTION_BUILDING[action_id]
        if _researching_at(s, building) >= max(s.buildings.get(building, 0), 1):
            return f"{building} is busy researching"
    if action_id == 59:
        return "not implemented by the bot"

    # --- scouting (handle_scouting): dynamic rules send one scout at a time ---
    if not legacy and action_id in (60, 61, 62, 63) and s.scouts_out:
        return "a scout is already out"

    # --- chronoboost needs a busy target and a Nexus with 50 energy (apply_chronoboost) ---
    if action_id in CHRONO and s.chrono_ready is False:
        return "no Nexus has 50 energy"
    if action_id in CHRONO:
        building, supply_req = CHRONO[action_id]
        if s.supply_left < supply_req:
            return f"the bot requires {supply_req} free supply for this chrono boost"
        busy = {
            "nexus": s.units_pending.get("probe", 0) > 0,
            "stargate": _pending_units(s, STARGATE_UNITS) > 0,
        }.get(building, _researching_at(s, building) > 0)
        if not busy:
            return f"{building} is idle"

    # --- army orders ---
    if legacy and action_id == 64 and s.army_supply < 10:
        return "attack() only moves out with at least 10 army supply (legacy rules)"
    return None
