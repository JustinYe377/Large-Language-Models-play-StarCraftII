import random

import numpy as np
import pytest

from sc2_rl_agent.starcraftenv_test.telemetry.funnel import (
    FunnelCounter, build_tick, fail_cat, match_failure, mix_with_sources, resolve_slot, source_labels,
)
from sc2_rl_agent.starcraftenv_test.utils.action_extractor import ActionExtractor, extract_actions_from_command
from sc2_rl_agent.starcraftenv_test.utils.action_info import ActionDescriptions

EMPTY = 71  # Protoss EMPTY ACTION


def legacy_mix_actions(real_actions, action_window, action_mix_rate, empty_action):
    """Verbatim copy of ChatGPTAgent.mix_actions before telemetry."""
    mixed_actions = []
    num_real_actions = int(action_window * action_mix_rate)
    if num_real_actions > len(real_actions):
        num_real_actions = len(real_actions)
    real_action_indices = np.linspace(start=0, stop=action_window - 1, num=num_real_actions, dtype=int)
    for i in range(action_window):
        if i in real_action_indices and real_actions:
            mixed_actions.append(real_actions.pop(0))
        else:
            mixed_actions.append(empty_action)
    return mixed_actions


def make_tick(proposed, kept, parse_tier="structured", k=5, window=10, rate=0.5, tick_id=0):
    sources = source_labels(proposed, kept, parse_tier, EMPTY)
    slots = mix_with_sources(list(kept), sources, window, rate, EMPTY)
    return build_tick(tick_id, 0, 0, 0.0, k, parse_tier, proposed, kept, slots, EMPTY), slots


def test_mix_with_sources_matches_legacy_placement():
    rng = random.Random(0)
    for _ in range(500):
        window = rng.choice([5, 8, 10, 12])
        rate = rng.choice([0.3, 0.5, 0.8, 1.0])
        actions = [rng.randrange(0, 72) for _ in range(rng.randrange(0, 9))]
        expected = legacy_mix_actions(list(actions), window, rate, EMPTY)
        got = mix_with_sources(list(actions), ["llm"] * len(actions), window, rate, EMPTY)
        assert [a for a, _ in got] == expected


def test_fake_tick_k5_parse4_filter1_window10():
    proposed = [0, 19, 22, 23]           # K=5 requested, 4 parsed
    kept = [0, 19, EMPTY, 23]            # resource filter dropped BUILD GATEWAY
    tick, slots = make_tick(proposed, kept)

    assert tick.n_parsed == 4
    assert tick.n_after_filter == 3
    assert tick.n_filter_dropped == 1
    assert tick.n_queued_real == 3
    assert tick.n_padding == 7
    assert tick.n_truncated == 0
    assert tick.queued_action_ids == [0, 19, 23]
    assert [s for _, s in slots].count("filter_drop") == 1
    assert [s for _, s in slots].count("padding") == 6

    fc = FunnelCounter()
    fc.add_tick(tick)
    r = fc.rolling()
    assert r["parse_rate"] == pytest.approx(0.8)
    assert r["admit_rate"] == pytest.approx(0.75)
    assert r["queue_rate"] == pytest.approx(1.0)
    assert r["padding_share"] == pytest.approx(0.7)


def test_truncation_counted():
    proposed = [0, 19, 22, 23, 24, 25, 26]     # LLM returned 7 actions, window keeps 5
    tick, _ = make_tick(proposed, list(proposed), k=5)
    assert tick.n_queued_real == 5
    assert tick.n_truncated == 2
    assert tick.n_queued_real <= 10 * 0.5


def test_parse_empty_tick():
    tick, slots = make_tick([EMPTY], [EMPTY], parse_tier="none")
    assert tick.n_parsed == 0
    assert tick.n_queued_real == 0
    assert tick.n_padding == 10
    assert [s for _, s in slots].count("parse_empty") == 1


def test_failure_matching_ignores_other_actions():
    failures = ["Action failed: CHRONOBOOST NEXUS, Reason: base is idle or already chronoboosted",
                "Action failed: BUILD PYLON, Reason: placement failed"]
    assert match_failure("TRAIN PROBE", failures) is None
    o = resolve_slot(0, "TRAIN PROBE", "llm", 0, 5, 30.0, "TRAIN PROBE", failures)
    assert not o.failed

    failures.append("Action failed: TRAIN PROBE, Reason: Cannot afford Probe")
    o = resolve_slot(0, "TRAIN PROBE", "llm", 0, 5, 30.0, "TRAIN PROBE", failures)
    assert o.failed
    assert o.fail_reason == "Cannot afford Probe"
    assert o.fail_cat == "cannot afford"


def test_empty_slot_never_fails():
    failures = ["Action failed: TRAIN PROBE, Reason: Cannot afford Probe"]
    o = resolve_slot(EMPTY, "EMPTY ACTION", "padding", 0, 1, 1.0, "EMPTY ACTION", failures)
    assert not o.failed


def test_whole_game_rates_exclude_padding():
    fc = FunnelCounter()
    tick, slots = make_tick([0, 19, 22, 23], [0, 19, EMPTY, 23])
    fc.add_tick(tick)
    names = {0: "TRAIN PROBE", 19: "BUILD PYLON", 23: "BUILD CYBERNETICSCORE", EMPTY: "EMPTY ACTION"}
    for step, (aid, src) in enumerate(slots):
        fails = ["Action failed: BUILD PYLON, Reason: placement failed"] if aid == 19 else []
        fc.add_outcome(resolve_slot(aid, names[aid], src, 0, step, float(step), names[aid], fails))
    t = fc.totals()
    assert t["llm_slots_dequeued"] == 3
    assert t["llm_slots_succeeded"] == 2
    assert t["exec_success"] == pytest.approx(2 / 3)
    assert t["conversion"] == pytest.approx(2 / 5)
    assert t["padding_share"] == pytest.approx(0.7)
    assert t["slots_padding"] == 6 and t["slots_filter_drop"] == 1


def test_fail_cat_buckets():
    assert fail_cat("Action failed: TRAIN PROBE, Reason: Cannot afford Probe") == "cannot afford"
    assert fail_cat("Action failed: X, Reason: base is idle or already chronoboosted") == "chrono timing"
    assert fail_cat("something else") == "other"


class _StubDB:
    def search_actions(self, action):
        return {"ids": ["19"], "documents": ["BUILD PYLON"]}


@pytest.mark.parametrize("command,tier", [
    ("blah\nACTIONS: 0, 19, 23", "structured"),
    ("1. Build a pylon (action 19: `BUILD PYLON`)\n2. Probe (action 0: `TRAIN PROBE`)", "id_based"),
    ("Decisions:\n1: <TRAIN PROBE>\n2: <BUILD PYLON>\n", "legacy"),
    ("Decisions:\n1: <BUILD A PYLON PLEASE>\n", "vectordb"),
    ("I have no idea.", "none"),
])
def test_extractor_reports_parse_tier_without_changing_result(command, tier):
    desc = ActionDescriptions("Protoss")
    extractor = ActionExtractor(desc.action_descriptions)
    info = {}
    with_info = extract_actions_from_command(command, extractor, desc.empty_action_id, _StubDB(), info=info)
    without = extract_actions_from_command(command, extractor, desc.empty_action_id, _StubDB())
    assert info["parse_tier"] == tier
    assert with_info == without
