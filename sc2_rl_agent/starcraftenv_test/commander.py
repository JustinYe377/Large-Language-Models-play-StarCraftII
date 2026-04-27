"""
Commander: a slow strategic-tier LLM controller that gates the existing
fast action-tier LLM.

Architecture:
    Commander  (fires every ~60 game seconds, OR on major events)
        -> emits: intent (text), saving_for (dict), forbidden_actions (list)
    Actor      (existing CoS pipeline, fires every decision step)
        -> reads commander's output, included in its prompt
        -> if it picks a forbidden action, we reject and re-ask (1x), then
           fall back to a safe default action.

This module is intentionally STANDALONE so you can run unit tests on it
without launching SC2. The bot integration is a thin adapter — see
`example_integration` at the bottom.

Wire-up checklist (do this in your worker, not here):
    1. Instantiate one Commander() per game.
    2. On every step, call commander.maybe_tick(game_state) — it decides
       whether enough game time has passed to re-plan.
    3. Get commander.intent / commander.saving_for / commander.forbidden_actions.
    4. Inject them into the actor's CoS prompt.
    5. After actor picks an action, call commander.validate(action) to confirm
       it's allowed; if not, re-prompt or fall back.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Commander state — what the strategic tier currently believes/wants
# ---------------------------------------------------------------------------

@dataclass
class CommanderState:
    intent: str = "Open standard. Build workers, scout, prepare for second nexus."
    saving_for: Dict[str, int] = field(default_factory=dict)   # e.g. {"minerals": 400, "purpose": "nexus"}
    forbidden_actions: List[str] = field(default_factory=list) # blacklist mode — used when no allowlist active

    # Allowlist mode: when set (non-empty), ONLY these actions are allowed.
    # The hard lock uses this in preference to forbidden_actions when active.
    # Useful for "save for Nexus" scenarios where blacklisting individual
    # spending actions is brittle (too many things cost minerals).
    allowed_actions: List[str] = field(default_factory=list)

    # Tech path commitment — once set to a value other than "none", it
    # persists across ticks and the parser refuses to change it. Prevents
    # tick-to-tick drift between Robotics / Twilight / Stargate.
    committed_tech_path: str = "none"  # one of: "none", "robotics", "twilight", "stargate"
    tech_committed_at_seconds: float = -1.0

    last_tick_game_seconds: float = -1e9                       # forces a tick on first call
    tick_count: int = 0
    raw_last_response: str = ""                                # for debugging

    def to_prompt_block(self) -> str:
        """Render the strategic state into a chunk to splice into the actor's prompt."""
        lines = ["[COMMANDER DIRECTIVE — RESPECT THESE WHEN PICKING ACTIONS]"]
        lines.append(f"Strategic intent: {self.intent}")
        if self.saving_for:
            purpose = self.saving_for.get("purpose", "the planned build")
            mins = self.saving_for.get("minerals", 0)
            gas = self.saving_for.get("gas", 0)
            lines.append(
                f"Currently saving for {purpose} (need ~{mins} minerals, ~{gas} gas). "
                f"Until then, prefer non-spending actions: build workers, scout, micro existing units."
            )
        if self.committed_tech_path != "none":
            lines.append(
                f"Tech path COMMITTED: {self.committed_tech_path}. "
                f"Do not propose buildings/units from other tech paths."
            )
        if self.allowed_actions:
            lines.append(
                "ALLOWLIST MODE — only these actions will be accepted this minute: "
                + ", ".join(self.allowed_actions)
            )
        elif self.forbidden_actions:
            lines.append(
                "Actions FORBIDDEN this minute (will be auto-rejected): "
                + ", ".join(self.forbidden_actions)
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commander prompt template
# ---------------------------------------------------------------------------

# Full Protoss action vocabulary, mirrored from
# sc2_rl_agent/starcraftenv_test/utils/action_info.py::init_protoss_actions().
# Keep this in sync if the bot's vocab changes.
PROTOSS_ACTION_VOCAB = [
    # TRAIN UNIT
    'TRAIN PROBE', 'TRAIN ZEALOT', 'TRAIN ADEPT', 'TRAIN STALKER',
    'TRAIN SENTRY', 'TRAIN HIGHTEMPLAR', 'TRAIN DARKTEMPLAR', 'TRAIN VOIDRAY',
    'TRAIN CARRIER', 'TRAIN TEMPEST', 'TRAIN ORACLE', 'TRAIN PHOENIX',
    'TRAIN MOTHERSHIP', 'TRAIN OBSERVER', 'TRAIN IMMORTAL',
    'TRAIN WARPPRISM', 'TRAIN COLOSSUS', 'TRAIN DISRUPTOR', 'MORPH ARCHON',
    # BUILD STRUCTURE
    'BUILD PYLON', 'BUILD ASSIMILATOR', 'BUILD NEXUS',
    'BUILD GATEWAY', 'BUILD CYBERNETICSCORE',
    'BUILD FORGE', 'BUILD TWILIGHTCOUNCIL',
    'BUILD ROBOTICSFACILITY', 'BUILD STARGATE', 'BUILD TEMPLARARCHIVE',
    'BUILD DARKSHRINE', 'BUILD ROBOTICSBAY',
    'BUILD FLEETBEACON', 'BUILD PHOTONCANNON', 'BUILD SHIELDBATTERY',
    # RESEARCH TECHNIQUE — long list, abbreviated; all included for validation
    'RESEARCH WARPGATERESEARCH', 'RESEARCH PROTOSSAIRWEAPONSLEVEL1',
    'RESEARCH PROTOSSAIRWEAPONSLEVEL2', 'RESEARCH PROTOSSAIRWEAPONSLEVEL3',
    'RESEARCH PROTOSSAIRARMORSLEVEL1', 'RESEARCH PROTOSSAIRARMORSLEVEL2',
    'RESEARCH PROTOSSAIRARMORSLEVEL3', 'RESEARCH ADEPTPIERCINGATTACK',
    'RESEARCH BLINKTECH', 'RESEARCH CHARGE',
    'RESEARCH PROTOSSGROUNDWEAPONSLEVEL1', 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL2',
    'RESEARCH PROTOSSGROUNDWEAPONSLEVEL3', 'RESEARCH PROTOSSGROUNDARMORSLEVEL1',
    'RESEARCH PROTOSSGROUNDARMORSLEVEL2', 'RESEARCH PROTOSSGROUNDARMORSLEVEL3',
    'RESEARCH PROTOSSSHIELDSLEVEL1', 'RESEARCH PROTOSSSHIELDSLEVEL2',
    'RESEARCH PROTOSSSHIELDSLEVEL3', 'RESEARCH EXTENDEDTHERMALLANCE',
    'RESEARCH GRAVITICDRIVE', 'RESEARCH OBSERVERGRAVITICBOOSTER',
    'RESEARCH PSISTORMTECH', 'RESEARCH VOIDRAYSPEEDUPGRADE',
    'RESEARCH PHOENIXRANGEUPGRADE', 'RESEARCH TEMPESTGROUNDATTACKUPGRADE',
    # OTHER ACTION
    'SCOUTING PROBE', 'SCOUTING OBSERVER', 'SCOUTING ZEALOT', 'SCOUTING PHOENIX',
    'MULTI-ATTACK', 'MULTI-RETREAT',
    'CHRONOBOOST NEXUS', 'CHRONOBOOST CYBERNETICSCORE',
    'CHRONOBOOST TWILIGHTCOUNCIL', 'CHRONOBOOST STARGATE', 'CHRONOBOOST FORGE',
    'EMPTY ACTION',
]

# Actions the Commander must NEVER forbid (econ/safety/no-op).
PROTOSS_NEVER_FORBID = {
    'TRAIN PROBE', 'BUILD PYLON', 'BUILD NEXUS', 'EMPTY ACTION',
    'SCOUTING PROBE', 'SCOUTING OBSERVER',
    'SCOUTING ZEALOT', 'SCOUTING PHOENIX',
    'CHRONOBOOST NEXUS',
}

# Allowlist used when saving_for is critical (e.g. second Nexus). When this
# allowlist is active, the actor can ONLY pick from these — every other
# spending action is implicitly blocked. This is more robust than a 5-item
# blacklist because there are 30+ ways to spend minerals in this vocab.
SAVING_FOR_ALLOWLIST = [
    'TRAIN PROBE',         # economy never hurts while saving
    'BUILD PYLON',          # supply must keep flowing
    'BUILD NEXUS',          # the saving target itself
    'BUILD ASSIMILATOR',    # gas is cheap and rarely competes with nexus money
    'SCOUTING PROBE',       # information is free
    'CHRONOBOOST NEXUS',    # speeds up exactly what we want
    'EMPTY ACTION',
]

# Tech-path classification — which actions belong to which tech tree.
# Used to enforce committed_tech_path. An action assigned to a path other
# than the committed one will be blocked.
TECH_PATH_ACTIONS: Dict[str, List[str]] = {
    "robotics": [
        "BUILD ROBOTICSFACILITY", "BUILD ROBOTICSBAY",
        "TRAIN OBSERVER", "TRAIN IMMORTAL", "TRAIN WARPPRISM",
        "TRAIN COLOSSUS", "TRAIN DISRUPTOR",
        "RESEARCH EXTENDEDTHERMALLANCE", "RESEARCH GRAVITICDRIVE",
        "RESEARCH OBSERVERGRAVITICBOOSTER",
    ],
    "twilight": [
        "BUILD TWILIGHTCOUNCIL", "BUILD TEMPLARARCHIVE", "BUILD DARKSHRINE",
        "TRAIN HIGHTEMPLAR", "TRAIN DARKTEMPLAR", "MORPH ARCHON",
        "RESEARCH BLINKTECH", "RESEARCH CHARGE", "RESEARCH ADEPTPIERCINGATTACK",
        "RESEARCH PSISTORMTECH",
        "CHRONOBOOST TWILIGHTCOUNCIL",
    ],
    "stargate": [
        "BUILD STARGATE", "BUILD FLEETBEACON",
        "TRAIN VOIDRAY", "TRAIN CARRIER", "TRAIN TEMPEST",
        "TRAIN ORACLE", "TRAIN PHOENIX", "TRAIN MOTHERSHIP",
        "RESEARCH PROTOSSAIRWEAPONSLEVEL1", "RESEARCH PROTOSSAIRWEAPONSLEVEL2",
        "RESEARCH PROTOSSAIRWEAPONSLEVEL3", "RESEARCH PROTOSSAIRARMORSLEVEL1",
        "RESEARCH PROTOSSAIRARMORSLEVEL2", "RESEARCH PROTOSSAIRARMORSLEVEL3",
        "RESEARCH VOIDRAYSPEEDUPGRADE", "RESEARCH PHOENIXRANGEUPGRADE",
        "RESEARCH TEMPESTGROUNDATTACKUPGRADE",
        "CHRONOBOOST STARGATE",
    ],
}

# Reverse map: action name -> tech path that owns it. Actions not in any
# tech path (gateway units, upgrades, econ) map to None and are unaffected
# by tech commitment.
ACTION_TECH_PATH: Dict[str, str] = {}
for _path, _actions in TECH_PATH_ACTIONS.items():
    for _a in _actions:
        ACTION_TECH_PATH[_a] = _path


COMMANDER_SYSTEM_PROMPT = """You are the strategic Commander for a Protoss agent playing StarCraft II (LotV)
against a built-in Zerg AI. You think once per game-minute. A faster sub-agent
picks individual actions; your job is to set strategic direction and prevent
the sub-agent from making short-sighted economic decisions.

You will be given a summary of the current game state. Respond with ONLY a
JSON object (no prose, no code fences) with these fields:

{
  "intent": "<one short sentence describing the next 60-90 seconds>",
  "saving_for": {
    "purpose": "<what we're saving for, or null>",
    "minerals": <int target threshold>,
    "gas": <int target threshold>
  },
  "use_allowlist": <true/false — see ALLOWLIST RULE below>,
  "tech_path": "<one of: none | robotics | twilight | stargate>",
  "forbidden_actions": ["<action names to forbid this minute>"],
  "reasoning": "<one sentence; not shown to actor>"
}

ALLOWLIST RULE: when saving_for is critical (especially 'second nexus') and
we are within ~150 minerals of the target, set "use_allowlist": true. The
runtime will then restrict the actor to a fixed safe set of actions
(probe/pylon/nexus/scout/gas/chrono/empty) until next tick. This is more
robust than enumerating forbidden actions one by one — there are 30+
spending actions and a 5-item blacklist will leak.

TECH_PATH RULE: once you commit to a tech path (robotics / twilight /
stargate), keep returning the SAME value on every subsequent tick. The
runtime FREEZES the first non-"none" value it sees and ignores any later
attempt to change it. Drift across ticks is the single biggest failure mode
this Commander is meant to fix; respect it.

== STANDARD PROTOSS vs ZERG (LotV, ladder-standard) ==

Phase 1: Opening (game time 0:00 -> ~3:30)
  Canonical Nexus-First build order (memorize):
    14 Probe -> 16 Pylon -> 17 Nexus (~1:50) -> 20 Gateway -> 20 Assimilator
    -> 26 Cybernetics Core -> 28 2nd Assimilator -> Warp Gate research starts
    -> ~3:30 Stalker x2 for defense
  Rule: SECOND NEXUS GOES DOWN BY 2:00. If we are past 2:30 and still on one
  base, the opening has failed and saving_for must be 'second nexus' immediately.
  A nexus that hasn't been planted by 4:30 looks like a one-base all-in to Zerg.

Phase 2: Tech-up and stabilize (3:30 -> 6:00)
  After Cyber Core: pick ONE tech path and commit. Switching paths mid-game wastes
  resources catastrophically. Options:
    * Twilight Council -> Blink Stalkers (mobile harass + timing attack at ~5:30)
    * Robotics -> Immortal/Observer (safer, detects burrow, hard-counters Roaches)
    * Stargate -> Oracle/Phoenix (denies Drones, scouts, transitions to air)
  Most common against built-in Zerg: Robotics into Immortals.

Phase 3: First push window (6:00 -> 8:30)
  Standard timing: Stalker/Immortal/Sentry push at ~8:00 off 2 saturated bases.
  Goal of the push: deny Zerg's third base, force defensive units instead of drones.
  Do NOT push without Immortals or critical mass. Do NOT push with floating minerals.

Phase 4: Third base and consolidation (after the push, ~9:00-11:00)
  Third Nexus goes down during or right after the push. Bank should drop to <300m.

== CRITICAL FAILURE MODES TO PREVENT ==

(F1) Nexus-greed trap. With 300-400 minerals banked, the sub-agent is tempted to
spend on a Gateway (150) or unit (~100) "while waiting." This delays the Nexus
by ~3 minutes of income, which compounds to >1500 lost minerals over the game.
RULE: when saving_for is 'second nexus' and minerals < 400, FORBID
'BUILD GATEWAY', 'BUILD CYBERNETICSCORE', 'BUILD FORGE', 'BUILD TWILIGHTCOUNCIL',
'BUILD ROBOTICSFACILITY', 'BUILD STARGATE', 'TRAIN ZEALOT', 'TRAIN STALKER',
'TRAIN SENTRY'. Allow only TRAIN PROBE, BUILD PYLON, BUILD NEXUS.

(F2) Tech-everything trap. The sub-agent sometimes builds Twilight + Robotics + Stargate
in the same game. Pick ONE tech path per game and forbid the others until late game.

(F3) Mineral float. If banked minerals exceed 500, we are under-producing.
saving_for should be empty and intent should explicitly mention spending the bank
(more gateways, more probes if not saturated, an expansion).

(F4) Supply block. If supply_used == supply_cap, building a Pylon is the only
correct action. Forbid everything else until supply unblocks.

== ACTION NAME VOCABULARY (CRITICAL — use these EXACT strings) ==

The sub-agent only understands these action names. forbidden_actions MUST use
the EXACT spelling and capitalization below, or the lock will silently fail.

Training units:
  TRAIN PROBE, TRAIN ZEALOT, TRAIN ADEPT, TRAIN STALKER, TRAIN SENTRY,
  TRAIN HIGHTEMPLAR, TRAIN DARKTEMPLAR, TRAIN VOIDRAY, TRAIN CARRIER,
  TRAIN TEMPEST, TRAIN ORACLE, TRAIN PHOENIX, TRAIN MOTHERSHIP,
  TRAIN OBSERVER, TRAIN IMMORTAL, TRAIN WARPPRISM, TRAIN COLOSSUS,
  TRAIN DISRUPTOR, MORPH ARCHON

Building structures:
  BUILD PYLON, BUILD ASSIMILATOR, BUILD NEXUS, BUILD GATEWAY,
  BUILD CYBERNETICSCORE, BUILD FORGE, BUILD TWILIGHTCOUNCIL,
  BUILD ROBOTICSFACILITY, BUILD STARGATE, BUILD TEMPLARARCHIVE,
  BUILD DARKSHRINE, BUILD ROBOTICSBAY, BUILD FLEETBEACON,
  BUILD PHOTONCANNON, BUILD SHIELDBATTERY

Other:
  SCOUTING PROBE, MULTI-ATTACK, MULTI-RETREAT,
  CHRONOBOOST NEXUS, CHRONOBOOST CYBERNETICSCORE, EMPTY ACTION

== OUTPUT RULES ==

* forbidden_actions: at most 5 items. Use the EXACT action names from the vocab above.
* NEVER forbid 'TRAIN PROBE', 'BUILD PYLON', 'BUILD NEXUS', 'EMPTY ACTION',
  or any 'SCOUTING ...' action.
* If unsure or game state is messy, prefer FEWER constraints. A wrong constraint
  is worse than no constraint.
* saving_for.minerals/gas should be the cost of the *single* thing you want next,
  not a long-term goal. e.g., second nexus = 400/0, twilight council = 150/100.
* Do not change tech path mid-game unless the previously committed tech building
  is destroyed.
"""


def build_commander_user_prompt(game_state_summary: str, prev_state: CommanderState) -> str:
    return (
        "Current game state:\n"
        f"{game_state_summary}\n\n"
        f"Your previous directive (for continuity): {prev_state.intent}\n\n"
        "Output the JSON object now."
    )


# ---------------------------------------------------------------------------
# Commander itself
# ---------------------------------------------------------------------------

# Default fallback values when LLM call fails or returns bad JSON.
SAFE_FALLBACK = CommanderState(
    intent="LLM commander unavailable — actor decides freely.",
    saving_for={},
    forbidden_actions=[],
)


@dataclass
class Commander:
    """
    Strategic-tier controller. Wraps an LLM call and a tick clock.

    The `llm_call` parameter is a function you supply: it takes
    (system_prompt, user_prompt) and returns a string response. Keeping it
    as a callable means this class has zero dependency on openai/deepseek/etc
    and is trivial to unit-test with a mock.
    """
    llm_call: Callable[[str, str], str]
    tick_interval_game_seconds: float = 60.0
    state: CommanderState = field(default_factory=CommanderState)

    # Allowed actions per the actor's vocabulary — used to validate that
    # the commander only forbids real actions. Set this in __post_init__
    # or pass at construction time. If empty, no validation is done.
    known_actions: List[str] = field(default_factory=list)

    # Actions the Commander is NEVER allowed to forbid, even if the LLM
    # tries. Hard guard against pathological directives.
    never_forbid: List[str] = field(default_factory=list)

    def maybe_tick(self, game_seconds: float, game_state_summary: str) -> bool:
        """
        Decide whether enough game time has passed to re-plan. If so, fire
        the LLM call and update self.state. Returns True if a tick happened.
        """
        if game_seconds - self.state.last_tick_game_seconds < self.tick_interval_game_seconds:
            return False
        self._tick(game_seconds, game_state_summary)
        return True

    def force_tick(self, game_seconds: float, game_state_summary: str) -> None:
        """Re-plan immediately regardless of clock — for use on major events."""
        self._tick(game_seconds, game_state_summary)

    def _tick(self, game_seconds: float, game_state_summary: str) -> None:
        prev = self.state
        try:
            response = self.llm_call(
                COMMANDER_SYSTEM_PROMPT,
                build_commander_user_prompt(game_state_summary, prev),
            )
            new_state = self._parse_response(response, game_seconds)
            new_state.tick_count = prev.tick_count + 1
            new_state.raw_last_response = response
            self.state = new_state
        except Exception as e:
            # Don't crash the game over a commander failure. Keep prior state,
            # bump the tick counter, but log the error.
            print(f"[Commander] LLM call failed at t={game_seconds:.0f}s: {e}")
            self.state.last_tick_game_seconds = game_seconds
            self.state.tick_count = prev.tick_count + 1

    def _parse_response(self, raw: str, game_seconds: float) -> CommanderState:
        """Robustly extract JSON. The LLM occasionally wraps it in fences or prose."""
        text = raw.strip()
        # Strip markdown fences if present.
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
        # Find the first '{' and last '}' as a fallback.
        if not text.startswith("{"):
            i, j = text.find("{"), text.rfind("}")
            if i == -1 or j == -1 or j < i:
                raise ValueError(f"No JSON object in response: {raw[:200]}")
            text = text[i : j + 1]

        data = json.loads(text)
        prev = self.state

        # ---- intent ----
        intent = str(data.get("intent", "")).strip() or prev.intent

        # ---- saving_for ----
        saving_for = data.get("saving_for") or {}
        if not isinstance(saving_for, dict):
            saving_for = {}

        # ---- forbidden_actions (blacklist) ----
        forbidden = data.get("forbidden_actions") or []
        if not isinstance(forbidden, list):
            forbidden = []
        forbidden = [str(a) for a in forbidden][:8]  # bumped from 5 — allowlist mode is the new primary control
        if self.known_actions:
            unknown = [a for a in forbidden if a not in self.known_actions]
            if unknown:
                print(f"[Commander] dropping unknown forbidden actions: {unknown}")
            forbidden = [a for a in forbidden if a in self.known_actions]
        if self.never_forbid:
            blocked = [a for a in forbidden if a in self.never_forbid]
            if blocked:
                print(f"[Commander] stripping never-forbid actions: {blocked}")
            forbidden = [a for a in forbidden if a not in self.never_forbid]

        # ---- allowlist mode ----
        # When use_allowlist is true, populate allowed_actions with the canonical
        # SAVING_FOR_ALLOWLIST. We do NOT let the LLM define an arbitrary
        # allowlist — that's too easy to mess up and turn into a softer lock.
        # Instead, allowlist mode is binary: on or off, with a fixed safe set.
        use_allowlist = bool(data.get("use_allowlist", False))
        allowed_actions: List[str] = []
        if use_allowlist:
            # Filter against known_actions so we never produce allowlist entries
            # the bot doesn't understand.
            if self.known_actions:
                allowed_actions = [a for a in SAVING_FOR_ALLOWLIST if a in self.known_actions]
            else:
                allowed_actions = list(SAVING_FOR_ALLOWLIST)

        # ---- tech_path commitment (sticky, server-enforced) ----
        proposed_path = str(data.get("tech_path", "none")).strip().lower()
        if proposed_path not in ("none", "robotics", "twilight", "stargate"):
            proposed_path = "none"

        if prev.committed_tech_path != "none":
            # Already committed — freeze. Ignore any LLM attempt to change.
            committed_tech_path = prev.committed_tech_path
            tech_committed_at_seconds = prev.tech_committed_at_seconds
            if proposed_path not in ("none", prev.committed_tech_path):
                print(f"[Commander] LLM tried to switch tech path "
                      f"{prev.committed_tech_path} -> {proposed_path}; ignoring")
        else:
            # Not yet committed — accept the LLM's first non-"none" choice.
            committed_tech_path = proposed_path
            tech_committed_at_seconds = (
                game_seconds if proposed_path != "none" else -1.0
            )
            if proposed_path != "none":
                print(f"[Commander] tech path COMMITTED: {proposed_path} at t={game_seconds:.0f}s")

        return CommanderState(
            intent=intent,
            saving_for=saving_for,
            forbidden_actions=forbidden,
            allowed_actions=allowed_actions,
            committed_tech_path=committed_tech_path,
            tech_committed_at_seconds=tech_committed_at_seconds,
            last_tick_game_seconds=game_seconds,
        )

    # ----- Hooks for the actor -----

    def get_prompt_block(self) -> str:
        return self.state.to_prompt_block()

    def is_allowed(self, action: str) -> bool:
        """
        Three-layer enforcement, checked in order:

          1. ALLOWLIST MODE: if state.allowed_actions is non-empty, the action
             must be IN that list. Everything else is rejected.
          2. TECH COMMITMENT: if a tech path is committed and this action
             belongs to a different tech path, reject.
          3. BLACKLIST: if the action is in state.forbidden_actions, reject.

        Otherwise allow.
        """
        s = self.state
        # 1. Allowlist takes precedence over everything else.
        if s.allowed_actions:
            return action in s.allowed_actions
        # 2. Tech path lock.
        if s.committed_tech_path != "none":
            owner = ACTION_TECH_PATH.get(action)
            if owner is not None and owner != s.committed_tech_path:
                return False
        # 3. Plain blacklist.
        return action not in s.forbidden_actions

    def validate(self, action: str, fallback: str = "TRAIN PROBE") -> str:
        """
        Convenience wrapper: returns the action if allowed, else returns
        the fallback.
        """
        return action if self.is_allowed(action) else fallback


# ---------------------------------------------------------------------------
# Standalone test harness — run this file directly
# ---------------------------------------------------------------------------

def _make_mock_llm(canned_response: str):
    """A deterministic 'LLM' for unit testing."""
    def _call(system: str, user: str) -> str:
        return canned_response
    return _call


def _self_test():
    print("=" * 60)
    print("Commander self-test")
    print("=" * 60)

    # ---- Test 1: happy path, the bug scenario from your replay ----
    # Uses the REAL action names from action_info.py.
    bug_response = json.dumps({
        "intent": "Save up for second nexus, defend with existing units.",
        "saving_for": {"purpose": "second nexus", "minerals": 400, "gas": 0},
        "forbidden_actions": ["BUILD GATEWAY", "BUILD CYBERNETICSCORE", "TRAIN ZEALOT"],
        "reasoning": "We are at 350m and committed to expand; spending elsewhere delays it.",
    })
    cmd = Commander(
        llm_call=_make_mock_llm(bug_response),
        tick_interval_game_seconds=60.0,
        known_actions=PROTOSS_ACTION_VOCAB,
        never_forbid=list(PROTOSS_NEVER_FORBID),
    )
    fired = cmd.maybe_tick(game_seconds=180.0, game_state_summary="t=3:00, 1 nexus, 14 workers, 350m/0g")
    assert fired, "first call should always tick"
    print("\nTest 1: bug scenario (real action names)")
    print(cmd.get_prompt_block())
    assert "second nexus" in cmd.state.saving_for.get("purpose", "")
    assert "BUILD GATEWAY" in cmd.state.forbidden_actions
    assert not cmd.is_allowed("BUILD GATEWAY")
    assert cmd.is_allowed("TRAIN PROBE")
    print("[ok] BUILD GATEWAY locked; TRAIN PROBE allowed")

    # ---- Test 2: tick interval respected ----
    fired_again = cmd.maybe_tick(game_seconds=200.0, game_state_summary="...")
    assert not fired_again, "should not tick before interval elapses"
    fired_late = cmd.maybe_tick(game_seconds=260.0, game_state_summary="...")
    assert fired_late, "should tick after interval elapses"
    print("\nTest 2: tick interval respected (60s game-time gating)")

    # ---- Test 3: malformed LLM response ----
    bad_cmd = Commander(llm_call=_make_mock_llm("uhh I dunno what to do"))
    bad_cmd._tick(60.0, "...")
    print("\nTest 3: malformed response -> graceful fallback")
    print(f"  intent kept: {bad_cmd.state.intent[:60]}")

    # ---- Test 4: response wrapped in code fences, with unknown action ----
    fenced = "```json\n" + json.dumps({
        "intent": "Tech to stargate.",
        "saving_for": {"purpose": "stargate", "minerals": 150, "gas": 150},
        "forbidden_actions": ["TRAIN ZEALOT", "made_up_action_name"],
        "reasoning": "Air opener.",
    }) + "\n```"
    fenced_cmd = Commander(
        llm_call=_make_mock_llm(fenced),
        known_actions=PROTOSS_ACTION_VOCAB,
        never_forbid=list(PROTOSS_NEVER_FORBID),
    )
    fenced_cmd.maybe_tick(0.0, "...")
    assert fenced_cmd.state.saving_for["purpose"] == "stargate"
    assert "TRAIN ZEALOT" in fenced_cmd.state.forbidden_actions
    assert "made_up_action_name" not in fenced_cmd.state.forbidden_actions
    print("\nTest 4: fenced response parsed; unknown action dropped")

    # ---- Test 5: never_forbid guard (LLM tries to forbid TRAIN PROBE) ----
    bad_directive = json.dumps({
        "intent": "Bad directive that would lock TRAIN PROBE.",
        "saving_for": {"purpose": "second nexus", "minerals": 400, "gas": 0},
        "forbidden_actions": ["TRAIN PROBE", "BUILD GATEWAY"],
        "reasoning": "...",
    })
    guarded_cmd = Commander(
        llm_call=_make_mock_llm(bad_directive),
        known_actions=PROTOSS_ACTION_VOCAB,
        never_forbid=list(PROTOSS_NEVER_FORBID),
    )
    guarded_cmd.maybe_tick(0.0, "...")
    assert "TRAIN PROBE" not in guarded_cmd.state.forbidden_actions
    assert "BUILD GATEWAY" in guarded_cmd.state.forbidden_actions
    print("\nTest 5: never_forbid guard works (TRAIN PROBE stripped)")

    # ---- Test 6: validate() helper with real fallback ----
    print("\nTest 6: validate() helper")
    print(f"  BUILD GATEWAY -> {cmd.validate('BUILD GATEWAY')}  (forbidden, falls back to TRAIN PROBE)")
    print(f"  TRAIN PROBE   -> {cmd.validate('TRAIN PROBE')}    (allowed)")

    # ---- Test 7: allowlist mode locks down to safe actions ----
    allowlist_response = json.dumps({
        "intent": "Lock everything down — saving for nexus.",
        "saving_for": {"purpose": "second nexus", "minerals": 400, "gas": 0},
        "use_allowlist": True,
        "tech_path": "none",
        "forbidden_actions": [],  # not used when allowlist active
        "reasoning": "Within 50 of target; minimize spending leakage.",
    })
    allow_cmd = Commander(
        llm_call=_make_mock_llm(allowlist_response),
        known_actions=PROTOSS_ACTION_VOCAB,
        never_forbid=list(PROTOSS_NEVER_FORBID),
    )
    allow_cmd.maybe_tick(0.0, "...")
    assert allow_cmd.state.allowed_actions, "allowlist should be populated"
    assert allow_cmd.is_allowed("TRAIN PROBE"), "probe must be allowed"
    assert allow_cmd.is_allowed("BUILD NEXUS"), "the saving target must be allowed"
    assert allow_cmd.is_allowed("BUILD ASSIMILATOR"), "gas building should be allowed"
    # All these would have leaked through a 5-item blacklist; allowlist catches them all.
    assert not allow_cmd.is_allowed("BUILD GATEWAY")
    assert not allow_cmd.is_allowed("BUILD CYBERNETICSCORE")
    assert not allow_cmd.is_allowed("BUILD FORGE")
    assert not allow_cmd.is_allowed("BUILD PHOTONCANNON")
    assert not allow_cmd.is_allowed("RESEARCH WARPGATERESEARCH")
    assert not allow_cmd.is_allowed("TRAIN STALKER")
    print("\nTest 7: allowlist mode blocks ALL spending leaks (gateway, forge, cannon, research, stalker)")

    # ---- Test 8: tech path commits and freezes ----
    tech_first_tick = json.dumps({
        "intent": "Going Robotics for Immortals.",
        "saving_for": {},
        "use_allowlist": False,
        "tech_path": "robotics",
        "forbidden_actions": [],
        "reasoning": "Robo for Immortals.",
    })
    tech_cmd = Commander(
        llm_call=_make_mock_llm(tech_first_tick),
        known_actions=PROTOSS_ACTION_VOCAB,
        never_forbid=list(PROTOSS_NEVER_FORBID),
    )
    tech_cmd.maybe_tick(120.0, "...")
    assert tech_cmd.state.committed_tech_path == "robotics"
    # A robotics action is allowed; a stargate action is blocked by tech lock.
    assert tech_cmd.is_allowed("BUILD ROBOTICSFACILITY")
    assert tech_cmd.is_allowed("TRAIN IMMORTAL")
    assert not tech_cmd.is_allowed("BUILD STARGATE"), "stargate should be locked under robotics commitment"
    assert not tech_cmd.is_allowed("TRAIN VOIDRAY"), "voidray should be locked under robotics commitment"
    # Gateway-tier actions belong to no tech path and remain unaffected.
    assert tech_cmd.is_allowed("TRAIN STALKER"), "gateway units have no tech ownership"
    assert tech_cmd.is_allowed("BUILD GATEWAY"), "gateway has no tech ownership"
    print("\nTest 8: tech path commits to robotics; stargate path locked out")

    # ---- Test 9: tech path is sticky — LLM cannot drift ----
    tech_drift = json.dumps({
        "intent": "Actually let's go stargate now.",
        "saving_for": {},
        "use_allowlist": False,
        "tech_path": "stargate",   # LLM trying to switch
        "forbidden_actions": [],
        "reasoning": "Reconsidered.",
    })
    # Use the same Commander; advance time and inject the drift attempt.
    tech_cmd.llm_call = _make_mock_llm(tech_drift)
    tech_cmd.maybe_tick(180.0, "...")
    assert tech_cmd.state.committed_tech_path == "robotics", \
        "drift to stargate should be ignored — once committed, robotics is sticky"
    print("\nTest 9: tech drift attempt rejected (stays robotics)")

    print("\nAll self-tests passed.")


# ---------------------------------------------------------------------------
# Example integration sketch — adapt this into your worker
# ---------------------------------------------------------------------------

EXAMPLE_INTEGRATION = '''
# In your bot / worker (pseudocode — adapt to your codebase):

import openai
from sc2_rl_agent.starcraftenv_test.commander import (
    Commander, PROTOSS_ACTION_VOCAB, PROTOSS_NEVER_FORBID,
)

def deepseek_call(system: str, user: str) -> str:
    """Bridge from Commander's callable interface to your LLM client."""
    resp = openai.ChatCompletion.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return resp["choices"][0]["message"]["content"]

# Once per game:
commander = Commander(
    llm_call=deepseek_call,
    tick_interval_game_seconds=60.0,
    known_actions=PROTOSS_ACTION_VOCAB,
    never_forbid=list(PROTOSS_NEVER_FORBID),
)

# Inside your bot's on_step (or wherever the actor decides):
async def on_step(self, iteration):
    game_seconds = self.time   # python-sc2 gives this as float seconds
    summary = build_state_summary(self)   # your existing CoS summary

    commander.maybe_tick(game_seconds, summary)

    # Inject commander directive into the actor's prompt:
    actor_prompt = (
        commander.get_prompt_block()
        + "\\n\\n"
        + your_existing_actor_prompt(summary)
    )

    action = call_actor_llm(actor_prompt)

    # Enforce the forbidden list. Two options:
    # (a) re-prompt once, then fall back:
    if not commander.is_allowed(action):
        action = call_actor_llm(actor_prompt + "\\nThe previous choice was forbidden. Pick another.")
        if not commander.is_allowed(action):
            action = "TRAIN PROBE"  # safe default — economy never hurts
    # (b) or just hard-fall-back (cheaper, faster):
    # action = commander.validate(action, fallback="build_probe")

    await execute_action(action)
'''


if __name__ == "__main__":
    _self_test()
    print("\n" + "=" * 60)
    print("Integration sketch — copy/adapt into your worker:")
    print("=" * 60)
    print(EXAMPLE_INTEGRATION)