"""
Event schemas for Phase-0 telemetry.

Every JSONL line written by RunLogger carries `schema_version`, `type` and
`ts` on top of the dataclass fields below. Bump SCHEMA_VERSION whenever a
field is renamed or its meaning changes, so old runs stay comparable.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional

SCHEMA_VERSION = 1

# Where a queue slot came from. Only `llm` counts as a real decision; the
# rest are filler that the funnel keeps out of the conversion denominator.
#   llm         - an action the LLM proposed that survived parsing + filter
#   padding     - EMPTY inserted by mix_actions between real actions
#   filter_drop - LLM proposal replaced by EMPTY by the resource filter
#   parse_empty - EMPTY placeholder because the extractor found nothing
#   queue_empty - EMPTY returned because the action queue was empty
SLOT_SOURCES = ("llm", "padding", "filter_drop", "parse_empty", "queue_empty")

# Which extractor tier produced the action ids (see utils/action_extractor.py).
PARSE_TIERS = ("structured", "id_based", "legacy", "vectordb", "none")

# SC2 "faster" speed: 22.4 game loops per in-game second.
GAME_LOOPS_PER_SECOND = 22.4

EMPTY_ACTION_NAME = "EMPTY ACTION"
ATTACK_ACTION_NAME = "MULTI-ATTACK"


@dataclass
class LLMCallRecord:
    """One HTTP attempt. Retries of the same logical call share `call_id`
    and are distinguished by `retries` (0 = first attempt)."""
    call_id: int
    game_step: int
    game_time_s: float
    purpose: str
    model_requested: str
    model_returned: Optional[str]
    reasoning_effort: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: Optional[int]
    cache_hit_tokens: Optional[int]
    cache_miss_tokens: Optional[int]
    latency_s: float
    retries: int
    error: Optional[str]
    finish_reason: Optional[str]
    output_chars: int
    has_reasoning_content: bool
    # True on the last attempt of a call that exhausted all retries (the
    # caller then receives the canned apology string instead of a command).
    final_failure: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class DecisionTick:
    """One L2 decision: LLM command -> parsed ids -> filter -> queue."""
    tick_id: int
    call_id: Optional[int]
    game_step: int
    game_time_s: float
    k_requested: int
    n_parsed: int
    parse_tier: str
    n_unique_parsed: int
    n_after_filter: int
    n_filter_dropped: int
    n_queued_real: int
    # Every non-`llm` slot in this tick's window (mix padding + filter_drop
    # + parse_empty). Padding never counts as a decision.
    n_padding: int
    proposed_action_ids: List[int]
    queued_action_ids: List[int]
    n_truncated: int = 0
    action_window: int = 0
    llm_error: bool = False
    slot_sources: List[str] = field(default_factory=list)
    command_preview: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class SlotOutcome:
    """What happened to one dequeued slot, read from the next observation."""
    slot_source: str
    action_id: int
    action_name: str
    executed_name: Optional[str]
    failed: bool
    fail_reason: Optional[str]
    fail_cat: Optional[str]
    tick_id: Optional[int] = None
    # Step whose action() call returned this slot to the env. Its result
    # shows up one step later, i.e. on line `dequeue_step + 1` of
    # action_executed.json / action_failures.json.
    dequeue_step: int = -1
    game_time_s: float = 0.0
    # False when the game ended before the result came back.
    resolved: bool = True

    def to_dict(self):
        return asdict(self)
