"""
Backfill on a hand-built 20-step legacy game folder.

Tick 0 (step 0):  "ACTIONS: 0, 19, 22, 23, 1" -> 5 real slots at positions [0, 2, 4, 6, 9]
Tick 1 (step 10): "ACTIONS: 0, 22"            -> 2 slots at [0, 9]; BUILD GATEWAY was
                  rejected by the resource filter -> 1 llm + 1 filter_drop
Dequeued slots: 20 = 6 llm + 1 filter_drop + 13 padding
  -> padding_share = 13 / 20, nonreal_share = 14 / 20.
"""

import json
import os

import pytest

from sc2_rl_agent.starcraftenv_test.telemetry.backfill import backfill_game, replay_info

E = "EMPTY ACTION"
SLOT_NAMES = (["TRAIN PROBE", E, "BUILD PYLON", E, "BUILD GATEWAY", E, "BUILD CYBERNETICSCORE", E, E,
               "TRAIN ZEALOT"] +
              ["TRAIN PROBE"] + [E] * 9)
REPLAY = ("20260101_000000_Ancient Cistern LE_player_race_PROTOSS_VS_BUILD_IN_AI_Harder_Zerg_"
          "process_-1_Result.Victory.SC2Replay")


def raw_obs(step):
    return {
        "resource": str({"game_time": f"00:{step:02d}", "worker_supply": 12 + step, "mineral": 50 + step,
                         "gas": 0, "supply_left": 4, "supply_cap": 23, "supply_used": 19,
                         "army_supply": 0}),
        "building": str({"nexus_count": 1, "pylon_count": 1, "gateway_count": 0}),
        "unit": str({"probe_count": 12, "Zealot_count": 0}),
        "planning": "{}", "research": "{}", "enemy": "{'unit': {}, 'structure': {}}",
    }


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def legacy_game(tmp_path):
    g = tmp_path / "chatgpt_log" / "game_20260101_000000_-1"
    g.mkdir(parents=True)
    # Line i = result of the slot dequeued at step i-1 (line 0 is the reset obs).
    executed = [[]] + SLOT_NAMES[:-1]
    failures = [[] for _ in range(20)]
    failures[3] = ["Action failed: BUILD PYLON, Reason: placement failed"]
    # A failure from another source (scripted defend) on the BUILD GATEWAY step.
    failures[5] = ["Action failed: MULTI-ATTACK, Reason: no army"]
    write_jsonl(g / "action_executed.json", executed)
    write_jsonl(g / "action_failures.json", failures)
    write_jsonl(g / "raw_observation.json", [raw_obs(i) for i in range(20)])
    write_jsonl(g / "combined_input.json", [{"L1_summaries": [[]]}] * 2)
    write_jsonl(g / "commander.json", [["Plan...\nACTIONS: 0, 19, 22, 23, 1"], ["ACTIONS: 0, 22"]])
    write_jsonl(g / "resource_rejections.json", [
        {"step": 10, "resources": {"minerals": 60, "gas": 0, "supply_left": 4},
         "rejections": [{"action": "BUILD GATEWAY", "reason": "need 150 minerals, have 60"}]}])
    (g / REPLAY).write_bytes(b"replay")
    return g


def test_backfill_infers_padding_and_funnel(legacy_game):
    s = backfill_game(str(legacy_game), mode="disabled")

    assert s["n_steps"] == 20
    assert s["n_ticks"] == 2
    assert s["total_slots"] == 20
    assert s["slots_llm"] == 6
    assert s["slots_filter_drop"] == 1
    assert s["slots_padding"] == 13
    assert s["padding_share"] == pytest.approx(13 / 20)
    assert s["nonreal_share"] == pytest.approx(14 / 20)
    assert s["parse_rate"] == pytest.approx(7 / 10)
    assert s["admit_rate"] == pytest.approx(6 / 7)
    # Only BUILD PYLON failed; the MULTI-ATTACK failure belongs to another source.
    assert s["exec_success"] == pytest.approx(5 / 6)
    assert s["conversion"] == pytest.approx(5 / 10)
    assert s["backfill_inconsistent_slots"] == 0
    assert s["result"] == "Victory"
    assert s["backfilled"] is True

    out = legacy_game / "backfill"
    assert (out / "game_summary.json").exists()
    ticks = [json.loads(l) for l in open(out / "decision_ticks.jsonl")]
    assert [t["n_queued_real"] for t in ticks] == [5, 1]
    assert [t["n_padding"] for t in ticks] == [5, 8]
    assert [t["n_filter_drop_slots"] for t in ticks] == [0, 1]
    assert [t["n_nonreal_slots"] for t in ticks] == [5, 9]
    run_start = json.loads(open(out / "events.jsonl").readline())
    assert run_start["config"]["backfilled"] is True
    assert run_start["config"]["padding_inferred"] is True
    assert run_start["config"]["difficulty"] == "Harder"


def test_backfill_is_rerunnable(legacy_game):
    backfill_game(str(legacy_game), mode="disabled")
    s = backfill_game(str(legacy_game), mode="disabled")
    assert s["n_ticks"] == 2
    lines = open(legacy_game / "backfill" / "decision_ticks.jsonl").read().splitlines()
    assert len(lines) == 2


def test_backfill_leaves_original_files_untouched(legacy_game):
    before = {p: open(legacy_game / p, "rb").read() for p in os.listdir(legacy_game)
              if p.endswith(".json")}
    backfill_game(str(legacy_game), mode="disabled")
    after = {p: open(legacy_game / p, "rb").read() for p in before}
    assert before == after


def test_replay_info_parses_filename():
    info = replay_info("/x/" + REPLAY)
    assert info["map"] == "Ancient Cistern LE"
    assert info["difficulty"] == "Harder"
    assert info["opposite_race"] == "Zerg"
    assert info["result"] == "Result.Victory"
