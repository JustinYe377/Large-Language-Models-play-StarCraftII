"""
Conversion funnel: LLM proposal -> parsed -> affordable -> queued -> executed.

Non-real slots (any slot_source other than `llm`: padding, filter_drop,
parse_empty, queue_empty) are counted per source but never enter a funnel
denominator. That is the main correction over the old analyzer, which
counted all of them as "LLM silent".
"""

import re
from collections import Counter, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .schema import DecisionTick, SlotOutcome, EMPTY_ACTION_NAME


def fail_cat(reason):
    """Bucket a failure string. Moved here from sc2_llm_analyzer.py, which
    now imports it, so both tools categorize failures the same way."""
    r = reason.lower()
    if "cannot afford"     in r: return "cannot afford"
    if "supply"            in r: return "supply blocked"
    if "idle" in r or "already chrono" in r: return "chrono timing"
    if "no available"      in r: return "building unavailable"
    if "from warp gate"    in r: return "warp gate confusion"
    if "placement"         in r: return "placement error"
    if "not idle" in r or "not the right" in r: return "building busy"
    return "other"


# ---------------------------------------------------------------------------
# Queue construction
# ---------------------------------------------------------------------------

def mix_with_sources(real_actions, sources, action_window, action_mix_rate, empty_action):
    """
    Same placement as ChatGPTAgent.mix_actions, but every slot is paired with
    its source label. `real_actions` is consumed (popped) like the original.

    Returns a list of (action_id, slot_source) of length action_window.
    """
    sources = list(sources)
    mixed_actions = []
    num_real_actions = int(action_window * action_mix_rate)
    if num_real_actions > len(real_actions):
        num_real_actions = len(real_actions)
    real_action_indices = np.linspace(start=0, stop=action_window - 1, num=num_real_actions, dtype=int)
    for i in range(action_window):
        if i in real_action_indices and real_actions:
            source = sources.pop(0) if sources else "llm"
            mixed_actions.append((real_actions.pop(0), source))
        else:
            mixed_actions.append((empty_action, "padding"))
    return mixed_actions


def source_labels(proposed_ids, kept_ids, parse_tier, empty_idx):
    """Label each entry of the post-filter list. The resource filter keeps
    list length and swaps rejected ids for EMPTY, so we compare positions."""
    if parse_tier == "none":
        return ["parse_empty"] * len(kept_ids)
    labels = []
    for i, kept in enumerate(kept_ids):
        proposed = proposed_ids[i] if i < len(proposed_ids) else kept
        if kept == empty_idx and proposed != empty_idx:
            labels.append("filter_drop")
        else:
            labels.append("llm")
    return labels


def build_tick(tick_id, call_id, game_step, game_time_s, k_requested, parse_tier,
               proposed_ids, kept_ids, slots, empty_idx, llm_error=False, command_preview=""):
    """Summarize one decision tick. `slots` is the (action_id, source) list
    produced by mix_with_sources."""
    proposed_ids = list(proposed_ids)
    n_parsed = 0 if parse_tier == "none" else len(proposed_ids)
    n_filter_dropped = 0
    if n_parsed:
        n_filter_dropped = sum(1 for p, k in zip(proposed_ids, kept_ids) if k == empty_idx and p != empty_idx)
    n_after_filter = n_parsed - n_filter_dropped
    queued_real = [aid for aid, src in slots if src == "llm"]
    n_queued_real = len(queued_real)
    by_source = Counter(src for _, src in slots)
    return DecisionTick(
        tick_id=tick_id,
        call_id=call_id,
        game_step=game_step,
        game_time_s=game_time_s,
        k_requested=k_requested,
        n_parsed=n_parsed,
        parse_tier=parse_tier,
        n_unique_parsed=len(set(proposed_ids)) if n_parsed else 0,
        n_after_filter=n_after_filter,
        n_filter_dropped=n_filter_dropped,
        n_queued_real=n_queued_real,
        n_padding=by_source["padding"],
        proposed_action_ids=proposed_ids if n_parsed else [],
        queued_action_ids=queued_real,
        n_filter_drop_slots=by_source["filter_drop"],
        n_parse_empty_slots=by_source["parse_empty"],
        n_nonreal_slots=len(slots) - n_queued_real,
        n_truncated=n_after_filter - n_queued_real,
        action_window=len(slots),
        llm_error=llm_error,
        slot_sources=[src for _, src in slots],
        command_preview=command_preview,
    )


# ---------------------------------------------------------------------------
# Execution outcome
# ---------------------------------------------------------------------------

_FAILURE_RE = re.compile(r"Action failed:\s*(.*?),\s*Reason:\s*(.*)", re.DOTALL)


def parse_failure(text):
    """'Action failed: TRAIN PROBE, Reason: Cannot afford Probe' -> (name, reason)."""
    m = _FAILURE_RE.match(str(text).strip())
    if not m:
        return None, str(text)
    return m.group(1).strip(), m.group(2).strip()


def match_failure(action_name, failures):
    """
    Return the failure string that belongs to `action_name`, or None.

    The failure list for a step also contains failures from other sources
    (e.g. the scripted defend()), so we match on the action name instead of
    treating any non-empty list as a failure of this slot.
    """
    if not failures or not action_name:
        return None
    target = action_name.strip().upper()
    for f in failures:
        name, _ = parse_failure(f)
        if name is not None and name.upper() == target:
            return str(f)
    return None


def resolve_slot(action_id, action_name, slot_source, tick_id, dequeue_step,
                 game_time_s, action_executed, action_failures):
    """Combine a dequeued slot with the next observation's feedback."""
    executed_name = action_executed if isinstance(action_executed, str) else None
    failures = action_failures if isinstance(action_failures, list) else []
    hit = None if action_name == EMPTY_ACTION_NAME else match_failure(action_name, failures)
    return SlotOutcome(
        slot_source=slot_source,
        action_id=action_id,
        action_name=action_name,
        executed_name=executed_name,
        failed=hit is not None,
        fail_reason=parse_failure(hit)[1] if hit else None,
        fail_cat=fail_cat(hit) if hit else None,
        tick_id=tick_id,
        dequeue_step=dequeue_step,
        game_time_s=game_time_s,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _ratio(num, den):
    return float(num) / den if den else 0.0


class FunnelCounter:
    """Accumulates ticks and slot outcomes; produces rolling and whole-game rates."""

    def __init__(self, rolling_ticks=10):
        self.rolling_ticks = rolling_ticks
        self.ticks: List[DecisionTick] = []
        self.slot_counts = Counter()          # dequeued slots by source
        self.parse_tiers = Counter()
        self.llm_resolved = 0
        self.llm_succeeded = 0
        self._per_tick_exec: Dict[int, List[int]] = {}   # tick_id -> [resolved, succeeded]
        self._recent = deque(maxlen=rolling_ticks)

    def add_tick(self, tick: DecisionTick):
        self.ticks.append(tick)
        self.parse_tiers[tick.parse_tier] += 1
        self._recent.append(tick)
        self._per_tick_exec.setdefault(tick.tick_id, [0, 0])

    def add_outcome(self, o: SlotOutcome):
        self.slot_counts[o.slot_source] += 1
        if o.slot_source != "llm" or not o.resolved:
            return
        self.llm_resolved += 1
        ok = not o.failed
        self.llm_succeeded += ok
        if o.tick_id in self._per_tick_exec:
            rec = self._per_tick_exec[o.tick_id]
            rec[0] += 1
            rec[1] += ok

    @staticmethod
    def _rates(ticks: Sequence[DecisionTick], resolved, succeeded, slot_counts=None):
        k = sum(t.k_requested for t in ticks)
        parsed = sum(t.n_parsed for t in ticks)
        after = sum(t.n_after_filter for t in ticks)
        queued = sum(t.n_queued_real for t in ticks)
        out = {
            "parse_rate": _ratio(parsed, k),
            "admit_rate": _ratio(after, max(parsed, 1)),
            "queue_rate": _ratio(queued, max(after, 1)),
            "exec_success": _ratio(succeeded, max(resolved, 1)),
            "conversion": _ratio(succeeded, k),
        }
        # padding_share: mix_actions filler only. nonreal_share: every slot
        # that is not an LLM decision (padding + filter_drop + parse_empty
        # + queue_empty).
        if slot_counts is not None:
            total = sum(slot_counts.values())
            out["padding_share"] = _ratio(slot_counts.get("padding", 0), total)
            out["nonreal_share"] = _ratio(total - slot_counts.get("llm", 0), total)
        else:
            window = sum(t.action_window for t in ticks)
            out["padding_share"] = _ratio(sum(t.n_padding for t in ticks), window)
            out["nonreal_share"] = _ratio(sum(t.n_nonreal_slots for t in ticks), window)
        return out

    def rolling(self):
        ticks = list(self._recent)
        resolved = sum(self._per_tick_exec.get(t.tick_id, [0, 0])[0] for t in ticks)
        succeeded = sum(self._per_tick_exec.get(t.tick_id, [0, 0])[1] for t in ticks)
        return self._rates(ticks, resolved, succeeded)

    def totals(self):
        out = self._rates(self.ticks, self.llm_resolved, self.llm_succeeded, self.slot_counts)
        out["n_ticks"] = len(self.ticks)
        out["llm_slots_dequeued"] = self.slot_counts.get("llm", 0)
        out["llm_slots_succeeded"] = self.llm_succeeded
        out["total_slots"] = sum(self.slot_counts.values())
        out["truncated_actions"] = sum(t.n_truncated for t in self.ticks)
        out["filter_dropped_actions"] = sum(t.n_filter_dropped for t in self.ticks)
        for src, n in self.slot_counts.items():
            out[f"slots_{src}"] = n
        for tier, n in self.parse_tiers.items():
            out[f"parse_tier_{tier}"] = n
        return out
