"""
Resource-based action filtering.

Goal: drop actions from the actor's proposed batch that we obviously can't
afford given current minerals/gas/supply, and surface the rejection reason
so the actor learns. Pure deterministic math; no LLM, no commander.

Why this matters: in real game logs the actor frequently proposes actions
costing 600+ minerals when we have 50, then those actions become EMPTY ACTION
in the queue. ~50% of the actor's proposed actions become EMPTY this way.
This filter cuts that to whatever actually exceeds resources at the moment
of decision.

Scope (intentionally narrow):
  * Costs are STATIC resource costs (minerals, gas, supply). We do NOT model:
    - Building prerequisites (no Cyber Core => can't train Stalker). The bot's
      env layer already rejects these and reports "Action failed: ...".
    - Production capacity (already-busy buildings).
    - Probe availability for construction.
    - Tech research prerequisites (e.g., Charge requires Twilight Council).
  * Resources come from the latest L1 observation. We don't track resources
    consumed across the action_window — the env tells us truth on next tick.

Integration: import ACTION_COSTS / can_afford / filter_by_resources and
call filter_by_resources() in chatgpt_agent.action() right after
extract_actions_from_command(). See bottom of file for the call shape.
"""

from __future__ import annotations
import ast
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cost table — Protoss, LotV ladder values.
# Format: action_name -> {"minerals": int, "gas": int, "supply": int}
# Supply is the supply COST (units increase used; buildings/research = 0).
# Source: Liquipedia Protoss unit/structure pages, current as of 2026.
# ---------------------------------------------------------------------------
PROTOSS_ACTION_COSTS: Dict[str, Dict[str, int]] = {
    # ----- Units -----
    'TRAIN PROBE':       {'minerals':  50, 'gas':   0, 'supply': 1},
    'TRAIN ZEALOT':      {'minerals': 100, 'gas':   0, 'supply': 2},
    'TRAIN ADEPT':       {'minerals': 100, 'gas':  25, 'supply': 2},
    'TRAIN STALKER':     {'minerals': 125, 'gas':  50, 'supply': 2},
    'TRAIN SENTRY':      {'minerals':  50, 'gas': 100, 'supply': 2},
    'TRAIN HIGHTEMPLAR': {'minerals':  50, 'gas': 150, 'supply': 2},
    'TRAIN DARKTEMPLAR': {'minerals': 125, 'gas': 125, 'supply': 2},
    'TRAIN VOIDRAY':     {'minerals': 200, 'gas': 150, 'supply': 4},
    'TRAIN CARRIER':     {'minerals': 350, 'gas': 250, 'supply': 6},
    'TRAIN TEMPEST':     {'minerals': 250, 'gas': 175, 'supply': 5},
    'TRAIN ORACLE':      {'minerals': 150, 'gas': 150, 'supply': 3},
    'TRAIN PHOENIX':     {'minerals': 150, 'gas': 100, 'supply': 2},
    'TRAIN MOTHERSHIP':  {'minerals': 400, 'gas': 400, 'supply': 8},
    'TRAIN OBSERVER':    {'minerals':  25, 'gas':  75, 'supply': 1},
    'TRAIN IMMORTAL':    {'minerals': 275, 'gas': 100, 'supply': 4},
    'TRAIN WARPPRISM':   {'minerals': 250, 'gas':   0, 'supply': 2},
    'TRAIN COLOSSUS':    {'minerals': 300, 'gas': 200, 'supply': 6},
    'TRAIN DISRUPTOR':   {'minerals': 150, 'gas': 150, 'supply': 4},
    'MORPH ARCHON':      {'minerals':   0, 'gas':   0, 'supply': 0},  # consumes 2 templar units, no resource cost

    # ----- Buildings -----
    'BUILD PYLON':            {'minerals': 100, 'gas':   0, 'supply': 0},
    'BUILD ASSIMILATOR':      {'minerals':  75, 'gas':   0, 'supply': 0},
    'BUILD NEXUS':            {'minerals': 400, 'gas':   0, 'supply': 0},
    'BUILD GATEWAY':          {'minerals': 150, 'gas':   0, 'supply': 0},
    'BUILD CYBERNETICSCORE':  {'minerals': 150, 'gas':   0, 'supply': 0},
    'BUILD FORGE':            {'minerals': 150, 'gas':   0, 'supply': 0},
    'BUILD TWILIGHTCOUNCIL':  {'minerals': 150, 'gas': 100, 'supply': 0},
    'BUILD ROBOTICSFACILITY': {'minerals': 150, 'gas': 100, 'supply': 0},
    'BUILD STARGATE':         {'minerals': 150, 'gas': 150, 'supply': 0},
    'BUILD TEMPLARARCHIVE':   {'minerals': 150, 'gas': 200, 'supply': 0},
    'BUILD DARKSHRINE':       {'minerals': 150, 'gas': 150, 'supply': 0},
    'BUILD ROBOTICSBAY':      {'minerals': 150, 'gas': 150, 'supply': 0},
    'BUILD FLEETBEACON':      {'minerals': 300, 'gas': 200, 'supply': 0},
    'BUILD PHOTONCANNON':     {'minerals': 150, 'gas':   0, 'supply': 0},
    'BUILD SHIELDBATTERY':    {'minerals': 100, 'gas':   0, 'supply': 0},

    # ----- Research -----
    'RESEARCH WARPGATERESEARCH':           {'minerals':  50, 'gas':  50, 'supply': 0},
    'RESEARCH PROTOSSAIRWEAPONSLEVEL1':    {'minerals': 100, 'gas': 100, 'supply': 0},
    'RESEARCH PROTOSSAIRWEAPONSLEVEL2':    {'minerals': 175, 'gas': 175, 'supply': 0},
    'RESEARCH PROTOSSAIRWEAPONSLEVEL3':    {'minerals': 250, 'gas': 250, 'supply': 0},
    'RESEARCH PROTOSSAIRARMORSLEVEL1':     {'minerals': 150, 'gas': 150, 'supply': 0},
    'RESEARCH PROTOSSAIRARMORSLEVEL2':     {'minerals': 225, 'gas': 225, 'supply': 0},
    'RESEARCH PROTOSSAIRARMORSLEVEL3':     {'minerals': 300, 'gas': 300, 'supply': 0},
    'RESEARCH ADEPTPIERCINGATTACK':        {'minerals': 100, 'gas': 100, 'supply': 0},
    'RESEARCH BLINKTECH':                  {'minerals': 150, 'gas': 150, 'supply': 0},
    'RESEARCH CHARGE':                     {'minerals': 100, 'gas': 100, 'supply': 0},
    'RESEARCH PROTOSSGROUNDWEAPONSLEVEL1': {'minerals': 100, 'gas': 100, 'supply': 0},
    'RESEARCH PROTOSSGROUNDWEAPONSLEVEL2': {'minerals': 150, 'gas': 150, 'supply': 0},
    'RESEARCH PROTOSSGROUNDWEAPONSLEVEL3': {'minerals': 200, 'gas': 200, 'supply': 0},
    'RESEARCH PROTOSSGROUNDARMORSLEVEL1':  {'minerals': 100, 'gas': 100, 'supply': 0},
    'RESEARCH PROTOSSGROUNDARMORSLEVEL2':  {'minerals': 150, 'gas': 150, 'supply': 0},
    'RESEARCH PROTOSSGROUNDARMORSLEVEL3':  {'minerals': 200, 'gas': 200, 'supply': 0},
    'RESEARCH PROTOSSSHIELDSLEVEL1':       {'minerals': 150, 'gas': 150, 'supply': 0},
    'RESEARCH PROTOSSSHIELDSLEVEL2':       {'minerals': 225, 'gas': 225, 'supply': 0},
    'RESEARCH PROTOSSSHIELDSLEVEL3':       {'minerals': 300, 'gas': 300, 'supply': 0},
    'RESEARCH EXTENDEDTHERMALLANCE':       {'minerals': 150, 'gas': 150, 'supply': 0},
    'RESEARCH GRAVITICDRIVE':              {'minerals': 100, 'gas': 100, 'supply': 0},
    'RESEARCH OBSERVERGRAVITICBOOSTER':    {'minerals': 100, 'gas': 100, 'supply': 0},
    'RESEARCH PSISTORMTECH':               {'minerals': 150, 'gas': 150, 'supply': 0},
    'RESEARCH VOIDRAYSPEEDUPGRADE':        {'minerals': 100, 'gas': 100, 'supply': 0},
    'RESEARCH PHOENIXRANGEUPGRADE':        {'minerals': 150, 'gas': 150, 'supply': 0},
    'RESEARCH TEMPESTGROUNDATTACKUPGRADE': {'minerals': 150, 'gas': 150, 'supply': 0},

    # ----- Free actions (always affordable) -----
    'SCOUTING PROBE':              {'minerals': 0, 'gas': 0, 'supply': 0},
    'SCOUTING OBSERVER':           {'minerals': 0, 'gas': 0, 'supply': 0},
    'SCOUTING ZEALOT':             {'minerals': 0, 'gas': 0, 'supply': 0},
    'SCOUTING PHOENIX':            {'minerals': 0, 'gas': 0, 'supply': 0},
    'MULTI-ATTACK':                {'minerals': 0, 'gas': 0, 'supply': 0},
    'MULTI-RETREAT':               {'minerals': 0, 'gas': 0, 'supply': 0},
    'CHRONOBOOST NEXUS':           {'minerals': 0, 'gas': 0, 'supply': 0},
    'CHRONOBOOST CYBERNETICSCORE': {'minerals': 0, 'gas': 0, 'supply': 0},
    'CHRONOBOOST TWILIGHTCOUNCIL': {'minerals': 0, 'gas': 0, 'supply': 0},
    'CHRONOBOOST STARGATE':        {'minerals': 0, 'gas': 0, 'supply': 0},
    'CHRONOBOOST FORGE':           {'minerals': 0, 'gas': 0, 'supply': 0},
    'EMPTY ACTION':                {'minerals': 0, 'gas': 0, 'supply': 0},
}


# ---------------------------------------------------------------------------
# Resource extraction from raw_observation
# ---------------------------------------------------------------------------

def parse_resources(raw_observation) -> Optional[Dict[str, int]]:
    """
    Extract minerals, gas, supply_left from raw_observation.

    The observation's 'resource' field is a STRINGIFIED dict (single quotes,
    Python repr style), not JSON, so we use ast.literal_eval. Returns None
    if the field is missing or unparseable so callers can skip filtering
    rather than blow up.

    Expected schema (from log inspection):
        raw_observation['resource'] == "{'game_time': '02:30',
            'worker_supply': 21, 'mineral': 195, 'gas': 236,
            'supply_left': 8, 'supply_cap': 47, 'supply_used': 39,
            'army_supply': 4}"
    """
    if not isinstance(raw_observation, dict):
        return None
    resource = raw_observation.get('resource')
    parsed = None
    if isinstance(resource, dict):
        parsed = resource
    elif isinstance(resource, str):
        try:
            parsed = ast.literal_eval(resource)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, dict):
        return None
    try:
        return {
            'minerals':    int(parsed.get('mineral', 0)),
            'gas':         int(parsed.get('gas', 0)),
            'supply_left': int(parsed.get('supply_left', 0)),
        }
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Affordability check
# ---------------------------------------------------------------------------

def can_afford(action_name: str, resources: Dict[str, int]) -> Tuple[bool, str]:
    """
    Return (affordable, reason). reason is empty string when affordable,
    otherwise a short human-readable explanation.

    If the action isn't in the cost table we conservatively allow it —
    better to let through an unknown action than block a real one due to
    a missing cost entry.
    """
    cost = PROTOSS_ACTION_COSTS.get(action_name)
    if cost is None:
        return True, ""

    if resources['minerals'] < cost['minerals']:
        return False, (
            f"need {cost['minerals']} minerals, have {resources['minerals']}"
        )
    if resources['gas'] < cost['gas']:
        return False, (
            f"need {cost['gas']} gas, have {resources['gas']}"
        )
    if cost['supply'] > 0 and resources['supply_left'] < cost['supply']:
        return False, (
            f"need {cost['supply']} supply, have {resources['supply_left']} free "
            f"(supply blocked)"
        )
    return True, ""


# ---------------------------------------------------------------------------
# Batch filter
# ---------------------------------------------------------------------------

def filter_by_resources(
    action_ids: List[int],
    id_to_name: Dict[int, str],
    resources: Dict[str, int],
    empty_action_idx: int,
    simulate_depletion: bool = False,
) -> Tuple[List[int], List[Tuple[str, str]]]:
    """
    Drop actions from the proposed batch that we obviously can't afford.

    Two modes:

    SNAPSHOT mode (simulate_depletion=False, the DEFAULT):
        Each action is checked against the SAME starting resource pool.
        We do not deduct cost across the batch.
        This is the safer default. It preserves the LLM's strategic
        priorities — if it lists BUILD PYLON before BUILD NEXUS, the
        filter does NOT artificially fail the Nexus just because the
        Pylon "would have" been spent first. The env will execute
        actions in queue order and fail any that legitimately run out
        of resources, which the actor learns about on the next tick
        via the existing "Action failed: cannot afford" feedback.

    SIMULATION mode (simulate_depletion=True):
        Walks the batch in order, deducting cost as we go. This is
        more pessimistic — it tends to reject expensive actions that
        come after cheap ones, even when the bot will accumulate
        income before reaching them.

    Why snapshot is usually right:
        The action_queue interleaves the actor's actions with EMPTY
        slots (mix_actions does this with action_mix_rate=0.5). Between
        slots, real game time passes and resources tick up. So the bot
        will often have MORE resources by the time action #2 fires than
        it had when action #1 fired. Simulation mode ignores income
        accrual and always assumes the bot is at decision-time minus
        all costs paid in zero time.

    Returns:
        kept_ids: same length as action_ids, with rejected entries replaced
                  by empty_action_idx
        rejections: list of (action_name, reason) for actions that were dropped
    """
    kept: List[int] = []
    rejections: List[Tuple[str, str]] = []
    sim = dict(resources)  # mutable copy used only when simulate_depletion=True

    for aid in action_ids:
        name = id_to_name.get(aid)
        if name is None:
            kept.append(aid)
            continue

        # Choose the resource snapshot to check against.
        check_against = sim if simulate_depletion else resources
        ok, reason = can_afford(name, check_against)

        if not ok:
            rejections.append((name, reason))
            kept.append(empty_action_idx)
            continue

        if simulate_depletion:
            cost = PROTOSS_ACTION_COSTS.get(name, {'minerals': 0, 'gas': 0, 'supply': 0})
            sim['minerals'] -= cost['minerals']
            sim['gas'] -= cost['gas']
            sim['supply_left'] -= cost['supply']
        kept.append(aid)

    return kept, rejections


def format_rejections_for_actor(rejections: List[Tuple[str, str]]) -> List[str]:
    """
    Format rejections in the same style as the env's existing 'failed_actions'
    log — that way the actor sees them in a familiar shape next tick:
        "Action failed: BUILD PYLON, Reason: need 100 minerals, have 75"
    """
    return [f"Action failed: {name}, Reason: {reason}" for name, reason in rejections]


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal id->name map matching action_info.py vocab.
    id_to_name = {
        0: 'TRAIN PROBE', 1: 'TRAIN ZEALOT', 3: 'TRAIN STALKER',
        19: 'BUILD PYLON', 21: 'BUILD NEXUS', 22: 'BUILD GATEWAY', 23: 'BUILD CYBERNETICSCORE',
        33: 'BUILD SHIELDBATTERY', 34: 'RESEARCH WARPGATERESEARCH',
        42: 'RESEARCH BLINKTECH',
        66: 'CHRONOBOOST NEXUS', 71: 'EMPTY ACTION',
    }
    EMPTY = 71

    print("=" * 60)
    print("Test 1: snapshot mode (default) on the BUILD NEXUS bug")
    print("=" * 60)
    # Scenario from a real game run: 260 minerals, 319 gas, 5 supply.
    # Actor proposed: BUILD PYLON, BUILD NEXUS, TRAIN PROBE, RESEARCH BLINK, CHRONO.
    # In simulation mode, BUILD NEXUS got rejected because BUILD PYLON ate
    # 100m first (260 - 100 = 160 < 400 nexus cost).
    # In snapshot mode, BUILD NEXUS still fails (260 < 400) — it's truly
    # unaffordable right now — but at least we're not penalizing it for
    # the Pylon. RESEARCH BLINK should pass (150m < 260m, 150g < 319g).
    res = {'minerals': 260, 'gas': 319, 'supply_left': 5}
    ids = [19, 21, 0, 42, 66]
    kept, rejs = filter_by_resources(ids, id_to_name, res, EMPTY)
    print(f"  Resources: {res}")
    print(f"  Proposed: {[id_to_name[i] for i in ids]}")
    print(f"  Kept:     {[id_to_name[i] for i in kept]}")
    print(f"  Rejected: {rejs}")
    expected_kept = [19, EMPTY, 0, 42, 66]  # only Nexus rejected (truly can't afford 400m)
    assert kept == expected_kept, f"expected {expected_kept}, got {kept}"
    assert len(rejs) == 1
    assert rejs[0][0] == 'BUILD NEXUS'
    print("  [ok] BUILD NEXUS correctly rejected for actually being unaffordable (260 < 400);")
    print("       RESEARCH BLINKTECH NOT rejected (snapshot mode doesn't deduct Pylon's cost)\n")

    print("=" * 60)
    print("Test 2: simulation mode (opt-in) on the same bug scenario")
    print("=" * 60)
    kept, rejs = filter_by_resources(ids, id_to_name, res, EMPTY, simulate_depletion=True)
    print(f"  Kept:     {[id_to_name[i] for i in kept]}")
    print(f"  Rejected: {[r[0] for r in rejs]}")
    # In sim mode: pylon eats 100, nexus fails (160<400), probe eats 50, blink fails (110<150),
    # chrono passes. So kept = [19, 71, 0, 71, 66], rejected = [NEXUS, BLINK].
    expected_kept_sim = [19, EMPTY, 0, EMPTY, 66]
    assert kept == expected_kept_sim, f"expected {expected_kept_sim}, got {kept}"
    print("  [ok] simulation mode is more aggressive — also rejects RESEARCH BLINKTECH\n")

    print("=" * 60)
    print("Test 3: snapshot mode on the early-game scenario")
    print("=" * 60)
    res2 = {'minerals': 195, 'gas': 236, 'supply_left': 8}
    ids2 = [1, 22, 34, 66, 33]  # zealot, gateway, warpgate research, chrono, shieldbattery
    kept, rejs = filter_by_resources(ids2, id_to_name, res2, EMPTY)
    print(f"  Resources: {res2}")
    print(f"  Proposed: {[id_to_name[i] for i in ids2]}")
    print(f"  Kept:     {[id_to_name[i] for i in kept]}")
    print(f"  Rejected: {[r[0] for r in rejs]}")
    # Snapshot: zealot 100m ok, gateway 150m ok (we have 195), warpgate 50m+50g ok,
    # chrono ok, shieldbattery 100m ok. ALL kept.
    expected_kept_snap = [1, 22, 34, 66, 33]
    assert kept == expected_kept_snap, f"expected {expected_kept_snap}, got {kept}"
    assert len(rejs) == 0
    print("  [ok] all 5 actions kept (each individually affordable from 195m/236g)\n")

    print("=" * 60)
    print("Test 4: empty resources — everything dropped in either mode")
    print("=" * 60)
    res3 = {'minerals': 0, 'gas': 0, 'supply_left': 0}
    kept, rejs = filter_by_resources([0, 19, 22], id_to_name, res3, EMPTY)
    assert kept == [EMPTY, EMPTY, EMPTY]
    assert len(rejs) == 3
    print(f"  Empty resources, snapshot: all 3 dropped, {len(rejs)} rejections")

    print()
    print("=" * 60)
    print("Test 5: free actions always kept regardless of resources")
    print("=" * 60)
    kept, rejs = filter_by_resources([66, 71], id_to_name, res3, EMPTY)
    assert kept == [66, 71]
    assert rejs == []
    print(f"  Chrono + Empty kept with 0 resources\n")

    print("=" * 60)
    print("Test 6: parse_resources on real log entry")
    print("=" * 60)
    real_obs = {
        'resource': "{'game_time': '02:30', 'worker_supply': 21, 'mineral': 195, "
                    "'gas': 236, 'supply_left': 8, 'supply_cap': 47, "
                    "'supply_used': 39, 'army_supply': 4}",
    }
    parsed = parse_resources(real_obs)
    assert parsed == {'minerals': 195, 'gas': 236, 'supply_left': 8}
    print(f"  Parsed: {parsed}\n")

    print("=" * 60)
    print("Test 7: format_rejections_for_actor formatting")
    print("=" * 60)
    sample_rejs = [('BUILD NEXUS', 'need 400 minerals, have 260')]
    feedback = format_rejections_for_actor(sample_rejs)
    print(f"  {feedback[0]}")
    assert feedback == ['Action failed: BUILD NEXUS, Reason: need 400 minerals, have 260']

    print("\nAll resource_filter self-tests passed.")