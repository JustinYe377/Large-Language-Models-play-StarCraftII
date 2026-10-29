"""
Prompt layout for the harness, ordered for prefix caching:

  system  (static for the whole game) role, rules, output contract, action
          catalog, strategy-skill index
  system  (changes only when a new skill is loaded) the active skill's text
  user    (dynamic) state snapshot, changes, last actions' outcomes, enemy
          seen, army-goal progress, legal ids, plan

Nothing time-dependent goes into the first system message, so every request
after the first can reuse the cached prefix (DeepSeek context caching, vLLM
prefix caching); the second one keeps the cache warm between skill switches.
"""

from typing import Dict, List, Optional

from .actions import catalog_text

SYSTEM_TEMPLATE = """You are the commander of a Protoss army in StarCraft II, playing against the built-in {opponent} AI.
You make one macro decision every few seconds of game time. A low-level bot executes your actions in order, one per game step, and reports what happened. The bot does not cast unit spells (no storm, force field, blink or disruptor shots); it only produces, builds, researches, chrono boosts, scouts, attacks and retreats.

How to decide:
- Choose up to {k} actions to queue now, most important first. Fewer is fine; an empty list means wait.
- Choose only ids listed as legal in this turn's message; anything else is discarded.
- Actions run over the next few seconds while income keeps coming in, so you may queue an action you can afford soon.
- Keep producing probes until about 22 per base, and turn spare resources into production, tech and army.
- Read the outcomes of your previous actions. Do not repeat an action that just failed for the same reason.
- Keep "plan" to one or two sentences: the strategy you are following. You will see it again next turn; change it only when the situation changes.
{skills_section}
Army goal: "army_goal" is the army you are building, usually taken from the active skill: mode "attack" or "defend", the target units [{{"unit", "count"}}], and attack_at_supply. Use mode "keep" to leave it unchanged. Its progress is shown every turn. Build the units in the goal instead of a bit of everything. In attack mode, order MULTI-ATTACK (64) once the goal is met; retreat (65) when the attack is clearly lost.

Answer only by submitting the decision as JSON with keys "plan" (string), "army_goal" (object) and "actions" (list of {{"id": int, "why": short string}}){load_field}.

Action catalog (id: name | cost | prerequisites):
{catalog}"""

SKILLS_SECTION = """
Strategy skills: playbooks for this matchup. Load one with load_skill when the game starts, after scouting shows what the enemy is building, or when the game moves to a new phase; it stays visible until you load another. Do not reload the active skill.
{index}
"""


def build_system_prompt(k: int, opponent: str = "", skill_index: str = "", json_load_field: bool = False) -> str:
    return SYSTEM_TEMPLATE.format(
        k=k, opponent=opponent or "computer", catalog=catalog_text(),
        skills_section=SKILLS_SECTION.format(index=skill_index) if skill_index else "",
        load_field=(', and "load_skill" (a skill name to read before deciding, or "")'
                    if skill_index and json_load_field else ""))


def active_skill_message(name: Optional[str], text: Optional[str]) -> Optional[dict]:
    if not name or not text:
        return None
    return {"role": "system", "content": f"Active strategy skill: {name}\n\n{text}"}


def _fmt(d: Dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items()) if v) or "nothing yet"


def goal_progress_lines(goal: Optional[dict], units: Dict[str, int], pending: Dict[str, int],
                        supply_used: int) -> List[str]:
    if not goal:
        return ["(none set)"]
    parts, met = [], True
    for unit, target in goal["units"].items():
        have = units.get(unit, 0)
        coming = pending.get(unit, 0)
        parts.append(f"{unit} {have}/{target}" + (f" (+{coming} in production)" if coming else ""))
        met = met and have >= target
    line = f"mode {goal['mode']}: " + (", ".join(parts) or "no units listed")
    lines = [line]
    if goal["mode"] == "attack":
        need = goal.get("attack_at_supply", 0)
        lines.append(f"attack at supply {need}; supply now {supply_used}")
        if met and supply_used >= need:
            lines.append("GOAL MET: order MULTI-ATTACK (64) now")
    return lines


def build_user_message(state_text: str, changes: List[str], outcomes: List[str],
                       legal_ids: List[int], plan: Optional[str], enemy_seen: Optional[Dict[str, int]] = None,
                       goal_lines: Optional[List[str]] = None, active_skill: Optional[str] = None) -> str:
    parts = ["## Current state", state_text]
    parts += ["", "## Changes since your last decision", "\n".join(f"- {c}" for c in changes) or "- (first decision)"]
    parts += ["", "## Results of your last actions", "\n".join(f"- {o}" for o in outcomes) or "- none yet"]
    if enemy_seen is not None:
        parts += ["", "## Enemy seen this game (most at once)", _fmt(enemy_seen)]
    if goal_lines is not None:
        parts += ["", "## Army goal", "\n".join(goal_lines)]
    if active_skill is not None:
        parts += ["", "## Active skill", active_skill or "(none loaded)"]
    parts += ["", "## Your current plan", plan or "(none yet)"]
    parts += ["", "## Legal action ids this turn", ", ".join(str(i) for i in legal_ids)]
    return "\n".join(parts)
