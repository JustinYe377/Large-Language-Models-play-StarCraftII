import json
import random
import socket
import threading
import types
from argparse import Namespace
from http.server import ThreadingHTTPServer

import pytest

from sc2_rl_agent.starcraftenv_test.harness.actions import PROTOSS_ACTIONS, legal_actions, catalog_text
from sc2_rl_agent.starcraftenv_test.harness.llm import (
    StructuredClient, parse_decision, thinking_extra_body, decision_schema,
)
from sc2_rl_agent.starcraftenv_test.harness.prompt import build_system_prompt, build_user_message
from sc2_rl_agent.starcraftenv_test.harness.state import parse_state, render_state, diff_states

N_IDS = 72


def raw_obs(t=0, minerals=50, gas=0, supply=(8, 13), buildings=None, units=None, pending=None,
            research=None):
    b = {"nexus_count": 1, "pylon_count": 0, "gas_buildings_count": 0, "gateway_count": 0,
         "cybernetics_core_count": 0, "statgate_count": 0, "forge_count": 0}
    b.update(buildings or {})
    u = {"probe_count": supply[0], "Zealot_count": 0, "high_templar_count": 0}
    u.update(units or {})
    m, s = divmod(t, 60)
    return {
        "resource": str({"game_time": f"{m:02d}:{s:02d}", "worker_supply": supply[0], "mineral": minerals,
                         "gas": gas, "supply_left": supply[1] - supply[0], "supply_cap": supply[1],
                         "supply_used": supply[0], "army_supply": 0}),
        "building": str(b),
        "unit": str(u),
        "planning": str({"planning_structure": pending or {}, "planning_unit": {}}),
        "research": str(research or {"cybernetics_core": {"warpgate_research_status": 0}}),
        "enemy": str({"unit": {}, "structure": {}}),
    }


# ---------------------------------------------------------------------------
# state + masking
# ---------------------------------------------------------------------------

def test_state_normalises_bot_key_names():
    s = parse_state(raw_obs(buildings={"statgate_count": 1, "gas_buildings_count": 2},
                            units={"Zealot_count": 3}, pending={"planning_pylon_count": 1}))
    assert s.buildings["stargate"] == 1 and s.buildings["assimilator"] == 2
    assert s.units["zealot"] == 3 and s.buildings_pending["pylon"] == 1


def test_research_progress_split():
    s = parse_state(raw_obs(research={"cybernetics_core": {"warpgate_research_status": 0.4},
                                      "forge": {"protoss_ground_weapon_level_1_research_status": 1}}))
    assert s.research_in_progress == {"warpgate": 0.4}
    assert s.research_done == ["protoss_ground_weapon_level_1"]


def test_opening_mask():
    # 8/13 supply: 5 free > the bot's pylon threshold of 3; nexus idle -> no chrono.
    legal = {PROTOSS_ACTIONS[i].name for i, r in legal_actions(parse_state(raw_obs())).items() if r is None}
    assert legal == {"TRAIN PROBE", "BUILD ASSIMILATOR", "BUILD NEXUS", "SCOUTING PROBE", "EMPTY ACTION"}


def test_mask_follows_tech_and_research():
    s = parse_state(raw_obs(buildings={"pylon_count": 1, "gateway_count": 1, "cybernetics_core_count": 1,
                                       "forge_count": 1},
                            research={"cybernetics_core": {"warpgate_research_status": 0.5},
                                      "forge": {"protoss_ground_weapon_level_1_research_status": 1}}))
    mask = legal_actions(s)
    assert mask[3] is None                                  # stalker unlocked by cyber core
    assert mask[34] == "already researching"                # warpgate
    assert mask[44] == "already researched"                 # ground weapons 1
    assert mask[45] == "needs twilight_council"             # ground weapons 2
    assert mask[16] == "needs robotics_facility and robotics_bay"
    assert mask[18] == "needs 2 high/dark templar"


def test_specs_match_action_vocabulary():
    assert len(PROTOSS_ACTIONS) == N_IDS
    assert len(catalog_text().splitlines()) == N_IDS


def test_diff_reports_changes_and_supply_block():
    a = parse_state(raw_obs(t=10))
    b = parse_state(raw_obs(t=40, supply=(13, 13), buildings={"pylon_count": 1}))
    changes = diff_states(a, b)
    assert "buildings: pylon +1" in changes and "SUPPLY BLOCKED" in changes
    assert diff_states(None, b) == []


# ---------------------------------------------------------------------------
# prompt layout: static prefix for caching
# ---------------------------------------------------------------------------

def test_system_prompt_is_static_and_user_message_carries_dynamic_parts():
    assert build_system_prompt(5, "VeryEasy Zerg") == build_system_prompt(5, "VeryEasy Zerg")
    s = parse_state(raw_obs(t=95, minerals=120))
    user = build_user_message(render_state(s), ["buildings: pylon +1"], ["TRAIN PROBE: done x2"], [0, 19, 71],
                              "expand early")
    assert "time 01:35" in user and "TRAIN PROBE: done x2" in user and "expand early" in user
    assert user.rstrip().endswith("0, 19, 71")
    assert "01:35" not in build_system_prompt(5, "VeryEasy Zerg")


# ---------------------------------------------------------------------------
# decision parsing + thinking switch
# ---------------------------------------------------------------------------

def test_parse_decision_variants():
    d, p = parse_decision('{"plan": "macro", "actions": [{"id": 0, "why": "eco"}, {"id": 19, "why": "supply"}]}', 5, N_IDS)
    assert d.action_ids == [0, 19] and d.plan == "macro" and p == []
    d, p = parse_decision('Sure!\n```json\n{"plan": "x", "actions": [{"id": "22", "why": ""}]}\n```', 5, N_IDS)
    assert d.action_ids == [22]
    d, p = parse_decision({"plan": "x", "actions": [{"id": 99}, {"id": 0}]}, 5, N_IDS)
    assert d.action_ids == [0] and "invalid action id 99" in p[0]
    d, p = parse_decision({"plan": "x", "actions": [{"id": i} for i in range(7)]}, 5, N_IDS)
    assert len(d.actions) == 5 and "too many" in p[0]
    d, p = parse_decision("no json here", 5, N_IDS)
    assert d is None


def test_thinking_switch_per_provider():
    assert thinking_extra_body("disabled", "auto", "https://api.deepseek.com/v1") == {"thinking": {"type": "disabled"}}
    assert thinking_extra_body("enabled", "auto", "http://gpu:8000/v1") == {"chat_template_kwargs": {"enable_thinking": True}}
    assert thinking_extra_body("auto", "auto", "https://api.deepseek.com/v1") == {}
    assert thinking_extra_body("disabled", "none", "x") == {}


# ---------------------------------------------------------------------------
# StructuredClient with a fake transport
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    from sc2_rl_agent.starcraftenv_test.harness import llm
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


class BadRequest(Exception):
    status_code = 400


def tool_response(args, model="m", name="submit_decision", extra_calls=()):
    def call(n, a, i):
        return types.SimpleNamespace(id=f"call_{i}", function=types.SimpleNamespace(
            name=n, arguments=json.dumps(a) if not isinstance(a, str) else a))
    calls = [call(n, a, i + 1) for i, (n, a) in enumerate(extra_calls)] + [call(name, args, 0)]
    msg = types.SimpleNamespace(content=None, tool_calls=calls)
    usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20, completion_tokens_details=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="tool_calls")],
                                 usage=usage, model=model)


def content_response(text):
    msg = types.SimpleNamespace(content=text, tool_calls=None)
    usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20, completion_tokens_details=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                 usage=usage, model="m")


class FakeTransport:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **req):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make_client(outcomes, mode="auto", base_url="https://api.deepseek.com/v1"):
    t = FakeTransport(outcomes)
    return StructuredClient("m", "k", base_url, 0.3, 5, N_IDS, mode=mode, client=t), t


def test_tools_mode_request_shape_and_cache_stable_schema():
    c, t = make_client([tool_response({"plan": "p", "actions": [{"id": 0, "why": "w"}]})])
    d, tier, payload, problems, _ = c.decide([{"role": "user", "content": "x"}])
    assert tier == "tools" and d.action_ids == [0]
    req = t.requests[0]
    assert req["tool_choice"]["function"]["name"] == "submit_decision"
    assert req["tools"][0]["function"]["parameters"] == decision_schema(5, N_IDS)
    assert req["extra_body"] == {"thinking": {"type": "disabled"}}


def test_downgrade_when_server_rejects_mode(monkeypatch):
    c, t = make_client([BadRequest("Error 400: tool use is not supported"),
                        BadRequest("400 This response_format type is unavailable now"),
                        content_response('{"plan": "p", "actions": [{"id": 19, "why": "w"}]}')])
    d, tier, _, _, _ = c.decide([{"role": "user", "content": "x"}])
    assert tier == "json_object" and d.action_ids == [19]
    assert c.mode == "json_object"                        # remembered for later calls
    assert t.requests[2]["response_format"] == {"type": "json_object"}


def test_repair_retry_then_success():
    c, t = make_client([tool_response("not json at all"),
                        tool_response({"plan": "p", "actions": [{"id": 22, "why": "w"}]})])
    d, tier, _, problems, _ = c.decide([{"role": "user", "content": "x"}])
    assert tier == "repair" and d.action_ids == [22]
    assert "not a valid decision" in t.requests[1]["messages"][-1]["content"]


def test_empty_action_list_is_a_valid_wait():
    c, _ = make_client([tool_response({"plan": "save for nexus", "actions": []})])
    d, tier, _, problems, _ = c.decide([{"role": "user", "content": "x"}])
    assert d is not None and d.actions == [] and tier == "tools" and problems == []


def test_transport_failure_returns_none(monkeypatch):
    from sc2_rl_agent.starcraftenv_test.harness import llm
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    c, _ = make_client([RuntimeError("down")] * 5)
    d, tier, payload, problems, _ = c.decide([{"role": "user", "content": "x"}])
    assert d is None and tier == "none"


# ---------------------------------------------------------------------------
# HarnessAgent end to end with a scripted transport (no SC2, no network)
# ---------------------------------------------------------------------------

def make_agent(tmp_path, outcomes):
    from sc2_rl_agent.starcraftenv_test.harness.agent import HarnessAgent
    from sc2_rl_agent.starcraftenv_test.utils.action_info import ActionDescriptions
    args = Namespace(player_race="Protoss", process_id="-1", current_time="20990101_000000",
                     agent_type="deepseek_harness", difficulty="VeryEasy", opposite_race="Zerg",
                     opposite_type="build_in", opposite_bot="x", wandb_mode="disabled", LLM_api_key="k",
                     LLM_model_name="m", log_root=str(tmp_path), harness_max_actions=5)
    agent = HarnessAgent("m", "k", "https://api.deepseek.com/v1", 0.3, args, ActionDescriptions("Protoss"))
    agent.llm_client.client = FakeTransport(outcomes)
    return agent


def obs(raw, executed=None, failures=None):
    return {"player_race": "Protoss", "opposite_race": "Zerg", "map_name": "x", "information": raw,
            "action_executed": executed if executed is not None else [], "action_failures": failures or [],
            "game_loop": 0}


def test_agent_masks_illegal_ids_and_feeds_back(tmp_path):
    agent = make_agent(tmp_path, [
        tool_response({"plan": "open gateway", "actions": [{"id": 0, "why": "eco"}, {"id": 3, "why": "army"},
                                                           {"id": 19, "why": "supply"}]}),
        tool_response({"plan": "", "actions": [{"id": 0, "why": "eco"}]}),
    ])
    queued = []
    for step in range(11):
        executed = agent._id_to_name.get(queued[-1]) if queued else []
        fails = ["Action failed: TRAIN PROBE, Reason: All Nexus are busy"] if step == 3 else []
        a, command, flag = agent.action(obs(raw_obs(t=step, minerals=200, supply=(12, 14)), executed, fails))
        queued.append(a)
    assert queued[0] == 0 and 3 not in queued              # stalker masked: no cybernetics core
    first = json.loads(open(f"{agent.game_folder}/combined_input.json").readline())
    second = json.loads(open(f"{agent.game_folder}/combined_input.json").readlines()[1])
    assert "L1_summaries" in first                         # analyzer-compatible legacy shape
    msg = second["harness_user_message"]
    assert "discarded 3 TRAIN STALKER: not legal (needs cybernetics_core)" in msg
    assert "TRAIN PROBE: FAILED x1 (All Nexus are busy)" in msg
    assert "open gateway" in msg                           # plan carried over an empty plan
    ticks = [json.loads(l) for l in open(f"{agent.game_folder}/decision_ticks.jsonl")]
    assert ticks[0]["parse_tier"] == "tools" and ticks[0]["n_parsed"] == 3
    assert ticks[0]["n_illegal_dropped"] == 1 and ticks[0]["n_after_filter"] == 2
    agent.telemetry.finish("aborted")


# ---------------------------------------------------------------------------
# Real HTTP against the mock server
# ---------------------------------------------------------------------------

def _serve(**kw):
    from sc2_rl_agent.starcraftenv_test.tests.mock_llm_server import make_handler
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(list(range(71)), 5, "mock", 0.0,
                                                                random.Random(0), **kw))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}/v1"


@pytest.mark.parametrize("reject_tools,expected_tier", [(False, "tools"), (True, "json_schema")])
def test_client_against_mock_server(reject_tools, expected_tier):
    srv, url = _serve(reject_tools=reject_tools)
    try:
        c = StructuredClient("mock", "x", url, 0.3, 5, N_IDS)
        user = build_user_message("state", [], [], [0, 19, 22], None)
        d, tier, _, problems, _ = c.decide([{"role": "user", "content": user}])
        assert tier == expected_tier and set(d.action_ids) <= {0, 19, 22} and d.plan
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# capacity masking (mirrors the bot's handlers)
# ---------------------------------------------------------------------------

def masked(raw, rules="dynamic"):
    return legal_actions(parse_state(raw), rules)


def test_probe_and_chrono_follow_nexus_activity():
    busy = raw_obs(supply=(12, 15))
    busy["planning"] = str({"planning_structure": {}, "planning_unit": {"planning_probe_count": 1}})
    m = masked(busy)
    assert m[0] is None                                    # bot v2 queues one probe behind the current one
    assert m[66] is None                                   # chrono is useful exactly now
    busy["planning"] = str({"planning_structure": {}, "planning_unit": {"planning_probe_count": 2}})
    assert masked(busy)[0] == "every nexus already has a probe queued"
    m = masked(raw_obs(supply=(12, 15)))
    assert m[0] is None and m[66] == "nexus is idle"
    assert masked(raw_obs(supply=(15, 15)))[0] == "no free supply"


@pytest.mark.parametrize("t,supply,pending,legal", [
    (100, (12, 15), 0, True),     # 3 free, threshold 3
    (100, (11, 15), 0, False),    # 4 free
    (100, (12, 15), 3, False),    # 3 pylons pending > 2
    (500, (10, 15), 0, True),     # 08:20: threshold rises to 7 free supply
])
def test_pylon_follows_legacy_thresholds(t, supply, pending, legal):
    raw = raw_obs(t=t, supply=supply)
    raw["planning"] = str({"planning_structure": {"planning_pylon_count": pending}, "planning_unit": {}})
    assert (masked(raw, "legacy")[19] is None) == legal


def test_dynamic_pylon_scales_with_production():
    # 1 nexus + 3 gateways + 1 robo: needs 2 + 2 + 6 + 4 = 14 free supply per cycle.
    b = {"pylon_count": 2, "gateway_count": 3, "cybernetics_core_count": 1, "robotics_facility_count": 1}
    assert masked(raw_obs(supply=(30, 40), buildings=b))[19] is None            # 10 free < 14
    assert "sufficient" in masked(raw_obs(supply=(30, 46), buildings=b))[19]    # 16 free
    pending = raw_obs(supply=(30, 40), buildings=b)
    pending["planning"] = str({"planning_structure": {"planning_pylon_count": 1}, "planning_unit": {}})
    assert "sufficient" in masked(pending)[19]                                  # 10 + 8 under way


def test_research_one_at_a_time_per_building():
    raw = raw_obs(buildings={"pylon_count": 1, "gateway_count": 1, "cybernetics_core_count": 1, "forge_count": 1},
                  research={"forge": {"protoss_ground_weapon_level_1_research_status": 0.3}})
    m = masked(raw)
    assert m[47] == "forge is busy researching" and m[50] == "forge is busy researching"
    assert m[34] is None                                    # cybernetics core still free
    assert m[70] is None and m[67] == "cybernetics_core is idle"


def test_gateway_units_after_warpgate():
    raw = raw_obs(buildings={"pylon_count": 1, "warp_gate_count": 2, "cybernetics_core_count": 1},
                  research={"cybernetics_core": {"warpgate_research_status": 1}}, supply=(20, 30))
    m = masked(raw)
    assert m[3] is None and m[1] is None and m[2] is None and m[4] is None   # bot v2 warps all of them
    # warp gate research done but the gateways have not transformed yet
    raw["building"] = raw["building"].replace("'warp_gate_count': 2", "'warp_gate_count': 0, 'gateway_count': 2")
    assert masked(raw)[3] == "no warp gate yet"


def test_building_limits_and_busy_production():
    raw = raw_obs(t=100, buildings={"pylon_count": 2, "gateway_count": 1, "cybernetics_core_count": 1,
                                    "robotics_facility_count": 1}, supply=(20, 30))
    raw["planning"] = str({"planning_structure": {}, "planning_unit": {"planning_immortal_count": 1}})
    m = masked(raw, "legacy")
    assert m[22] == "limit of 1 gateway until 03:00"
    assert masked(raw)[22] is None                          # no time-window limits under dynamic rules
    assert m[14] == "every robotics facility is busy"
    assert m[17] == "needs robotics_bay"                    # prerequisite reason comes first
    assert m[59] is not None                                # masked even once a fleet beacon exists
    beacon = raw_obs(buildings={"pylon_count": 1, "statgate_count": 1, "cybernetics_core_count": 1})
    beacon["building"] = beacon["building"][:-1] + ", 'fleet_beacon_count': 1}"
    assert masked(beacon)[59] == "not implemented by the bot"
    assert m[64] == "no army"
    small = raw_obs(supply=(20, 30))
    small["resource"] = small["resource"].replace("'army_supply': 0", "'army_supply': 6")
    assert masked(small, "legacy")[64].startswith("attack() only moves out")
    assert masked(small)[64] is None                        # dynamic: the agent decides when to attack


def test_cannon_and_battery_prerequisites_by_rules():
    forge_only = raw_obs(buildings={"pylon_count": 1, "forge_count": 1})
    assert masked(forge_only)[32] is None
    assert masked(forge_only, "legacy")[32] == "needs cybernetics_core (legacy rules)"
    core_only = raw_obs(buildings={"pylon_count": 1, "gateway_count": 1, "cybernetics_core_count": 1})
    assert masked(core_only)[33] is None
    assert masked(core_only, "legacy")[33] == "needs forge (legacy rules)"


def test_scout_and_chrono_masks_use_bot_extras():
    raw = raw_obs(supply=(12, 15))
    raw["planning"] = str({"planning_structure": {}, "planning_unit": {"planning_probe_count": 1}})
    m = legal_actions(parse_state(raw, {"scouts_out": 1, "chrono_ready": False}))
    assert m[60] == "a scout is already out" and m[66] == "no Nexus has 50 energy"
    m = legal_actions(parse_state(raw, {"scouts_out": 0, "chrono_ready": True}))
    assert m[60] is None and m[66] is None
    assert legal_actions(parse_state(raw, {"scouts_out": 1}), "legacy")[60] is None   # legacy uses its interval



# ---------------------------------------------------------------------------
# strategy skills, army goal, tool loop
# ---------------------------------------------------------------------------

from sc2_rl_agent.starcraftenv_test.harness.skills import SkillLibrary
from sc2_rl_agent.starcraftenv_test.harness.prompt import goal_progress_lines


def test_pvz_skill_library_loads():
    lib = SkillLibrary.for_opponent("Zerg")
    assert {"pvz-opening", "pvz-hold-early-rush", "pvz-vs-roach-ravager", "pvz-vs-ling-bane",
            "pvz-vs-hydra-lurker", "pvz-vs-mutalisk", "pvz-vs-air-late", "pvz-late-max"} <= set(lib.names)
    assert all(sk.description and sk.use_when and sk.body for sk in lib.skills.values())
    assert "pvz-vs-roach-ravager [mid" in lib.index_text()
    assert not SkillLibrary.for_opponent("Terran")           # only PvZ so far


def test_skill_tool_round_then_forced_submit():
    lib = SkillLibrary.for_opponent("Zerg")
    t = FakeTransport([
        tool_response({"name": "pvz-vs-roach-ravager"}, name="load_skill"),
        tool_response({"plan": "immortals", "army_goal": {"mode": "attack", "units": [{"unit": "immortal", "count": 6}],
                                                          "attack_at_supply": 120},
                       "actions": [{"id": 26, "why": "robo"}]}),
    ])
    c = StructuredClient("m", "k", "https://api.deepseek.com/v1", 0.3, 5, N_IDS, client=t, skills=lib)
    d, tier, _, problems, loaded = c.decide([{"role": "user", "content": "x"}])
    assert loaded == ["pvz-vs-roach-ravager"] and d.action_ids == [26]
    assert d.army_goal == {"mode": "attack", "units": {"immortal": 6}, "attack_at_supply": 120}
    first, second = t.requests
    assert first["tool_choice"] == "required" and [x["function"]["name"] for x in first["tools"]] == \
        ["submit_decision", "load_skill"]
    assert second["tools"] == first["tools"]                 # identical tool list keeps the cache
    assert second["tool_choice"]["function"]["name"] == "submit_decision"
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and "Immortals deal bonus damage" in tool_msg["content"]


def test_skill_and_decision_in_one_reply():
    lib = SkillLibrary.for_opponent("Zerg")
    t = FakeTransport([tool_response({"plan": "p", "army_goal": {"mode": "keep", "units": [], "attack_at_supply": 0},
                                      "actions": [{"id": 0, "why": "w"}]},
                                     extra_calls=[("load_skill", {"name": "pvz-opening"})])])
    c = StructuredClient("m", "k", "https://api.deepseek.com/v1", 0.3, 5, N_IDS, client=t, skills=lib)
    d, tier, _, _, loaded = c.decide([{"role": "user", "content": "x"}])
    assert loaded == ["pvz-opening"] and d.action_ids == [0] and d.army_goal is None and len(t.requests) == 1


def test_json_mode_load_skill_field():
    lib = SkillLibrary.for_opponent("Zerg")
    t = FakeTransport([content_response('{"plan": "", "army_goal": {"mode": "keep", "units": [], "attack_at_supply": 0},'
                                        ' "actions": [], "load_skill": "pvz-vs-mutalisk"}'),
                       content_response('{"plan": "phoenix", "army_goal": {"mode": "defend", "units": [{"unit": "phoenix",'
                                        ' "count": 6}], "attack_at_supply": 0}, "actions": [{"id": 27, "why": "sg"}],'
                                        ' "load_skill": ""}')])
    c = StructuredClient("m", "k", "http://gpu:8000/v1", 0.3, 5, N_IDS, mode="json_object", client=t, skills=lib)
    d, tier, _, _, loaded = c.decide([{"role": "user", "content": "x"}])
    assert loaded == ["pvz-vs-mutalisk"] and d.action_ids == [27] and tier == "json_object"
    assert "Phoenixes are faster" in t.requests[1]["messages"][-1]["content"]


def test_goal_progress_lines():
    goal = {"mode": "attack", "units": {"immortal": 2, "stalker": 4}, "attack_at_supply": 60}
    lines = goal_progress_lines(goal, {"immortal": 2, "stalker": 3}, {"stalker": 1}, 58)
    assert lines[0] == "mode attack: immortal 2/2, stalker 3/4 (+1 in production)"
    assert "GOAL MET" not in " ".join(lines)
    assert goal_progress_lines(goal, {"immortal": 2, "stalker": 4}, {}, 61)[-1].startswith("GOAL MET")
    assert goal_progress_lines(None, {}, {}, 0) == ["(none set)"]


def test_agent_keeps_active_skill_goal_and_enemy_memory(tmp_path):
    goal = {"mode": "attack", "units": [{"unit": "zealot", "count": 8}], "attack_at_supply": 40}
    agent = make_agent(tmp_path, [
        tool_response({"name": "pvz-vs-ling-bane"}, name="load_skill"),
        tool_response({"plan": "chargelots", "army_goal": goal, "actions": [{"id": 0, "why": "eco"}]}),
        tool_response({"plan": "", "army_goal": {"mode": "keep", "units": [], "attack_at_supply": 0},
                       "actions": [{"id": 0, "why": "eco"}]}),
    ])
    raw = raw_obs(t=30, minerals=200, supply=(12, 14))
    raw["enemy"] = str({"unit": {"enemy_UnitTypeId.ZERGLING": 8}, "structure": {}})
    for step in range(11):
        r = raw if step == 0 else raw_obs(t=30 + step, minerals=200, supply=(12, 14))
        agent.action(obs(r))
    assert agent.active_skill == "pvz-vs-ling-bane"
    assert agent.army_goal == {"mode": "attack", "units": {"zealot": 8}, "attack_at_supply": 40}
    second = agent.llm_client.client.requests[-1]
    assert second["messages"][1]["content"].startswith("Active strategy skill: pvz-vs-ling-bane")
    user = second["messages"][-1]["content"]
    assert "zergling 8" in user and "zealot 0/8" in user      # enemy memory survives the scout leaving
    events = [json.loads(l) for l in open(f"{agent.game_folder}/events.jsonl")]
    assert [e["type"] for e in events if e["type"] in ("skill_loaded", "army_goal_set")] == \
        ["skill_loaded", "army_goal_set"]
    s = agent.telemetry.finish("aborted")
    assert s["skills_loaded"] == ["pvz-vs-ling-bane"] and s["active_skill"] == "pvz-vs-ling-bane"
