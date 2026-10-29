"""
Prompt layout for the harness, ordered for prefix caching:

  system  (static for the whole game) role, rules, output contract, action catalog
  user    (dynamic) state snapshot, changes, last actions' outcomes, legal ids, plan

Nothing time-dependent goes into the system message, so every request after
the first can reuse the cached prefix (DeepSeek context caching, vLLM prefix
caching).
"""

from typing import Dict, List, Optional

from .actions import catalog_text

SYSTEM_TEMPLATE = """You are the commander of a Protoss army in StarCraft II, playing against the built-in {opponent} AI.
You make one macro decision every few seconds of game time. A low-level bot executes your actions in order, one per game step, and reports what happened.

How to decide:
- Choose up to {k} actions to queue now, most important first. Fewer is fine; an empty list means wait.
- Choose only ids listed as legal in this turn's message; anything else is discarded.
- Actions run over the next few seconds while income keeps coming in, so you may queue an action you can afford soon.
- Keep producing probes until about 22 per base, build pylons before you are supply blocked, and turn spare resources into production, tech and army.
- Read the outcomes of your previous actions. Do not repeat an action that just failed for the same reason.
- Chrono boost only helps a building that is busy producing or researching.
- Keep "plan" to one or two sentences: the strategy you are following. You will see it again next turn; change it only when the situation changes.

Answer only by submitting the decision as JSON with keys "plan" (string) and "actions" (list of {{"id": int, "why": short string}}).

Action catalog (id: name | cost | prerequisites):
{catalog}"""


def build_system_prompt(k: int, opponent: str = "") -> str:
    return SYSTEM_TEMPLATE.format(k=k, opponent=opponent or "computer", catalog=catalog_text())


def build_user_message(state_text: str, changes: List[str], outcomes: List[str],
                       legal_ids: List[int], plan: Optional[str]) -> str:
    parts = ["## Current state", state_text]
    parts += ["", "## Changes since your last decision", "\n".join(f"- {c}" for c in changes) or "- (first decision)"]
    parts += ["", "## Results of your last actions", "\n".join(f"- {o}" for o in outcomes) or "- none yet"]
    parts += ["", "## Your current plan", plan or "(none yet)"]
    parts += ["", "## Legal action ids this turn", ", ".join(str(i) for i in legal_ids)]
    return "\n".join(parts)
