import glob
import json
import os
import types
from argparse import Namespace

import pytest

from sc2_rl_agent.starcraftenv_test.telemetry.agent_hooks import AgentTelemetry
from sc2_rl_agent.starcraftenv_test.telemetry.funnel import build_tick, mix_with_sources, resolve_slot, source_labels
from sc2_rl_agent.starcraftenv_test.telemetry.llm_meter import LLMMeter
from sc2_rl_agent.starcraftenv_test.telemetry.run_logger import RunLogger, SUMMARY_FILE
from sc2_rl_agent.starcraftenv_test.telemetry.schema import LLMCallRecord, SCHEMA_VERSION

SECRET = "sk-test-SECRET-0123456789abcdef"
EMPTY = 71
NAMES = {0: "TRAIN PROBE", 19: "BUILD PYLON", 22: "BUILD GATEWAY", 23: "BUILD CYBERNETICSCORE",
         64: "MULTI-ATTACK", EMPTY: "EMPTY ACTION"}

SUMMARY_KEYS = [
    "result", "game_duration_s", "wall_clock_s", "n_steps", "n_ticks", "n_llm_calls", "n_llm_errors",
    "total_retries", "tokens_prompt", "tokens_completion", "tokens_reasoning", "cache_hit_tokens",
    "cache_miss_tokens", "cache_hit_ratio", "est_cost_usd", "latency_p50", "latency_p95",
    "wall_clock_share_waiting_llm", "parse_rate", "admit_rate", "queue_rate", "exec_success",
    "conversion", "padding_share", "nonreal_share", "peak_supply", "peak_workers", "max_bases",
    "time_to_first_attack_s", "scripted_attack_rule_steps", "scripted_attack_order_steps",
    "scripted_attack_units_ordered", "scripted_attack_first_order_s", "army_comp_entropy_mean",
    "supply_blocked_s", "model_resolved",
]


def make_args(**kw):
    base = dict(LLM_api_key=SECRET, LLM_model_name="deepseek-v4-flash", LLM_temperature=0.3,
                LLM_api_base="https://api.deepseek.com/v1", difficulty="VeryEasy",
                map_pool=["Altitude LE", "Ancient Cistern LE"], map_idx=1, player_race="Protoss",
                opposite_race="Zerg", opposite_type="build_in", agent_type="deepseek",
                current_time="20261023_120000", process_id="-1", exp_group="unit",
                price_in_hit=0.1, price_in_miss=1.0, price_out=2.0)
    base.update(kw)
    return Namespace(**base)


def raw_obs(t, minerals=100, supply_left=5, zealots=0, stalkers=0):
    m, s = divmod(int(t), 60)
    return {
        "resource": str({"game_time": f"{m:02d}:{s:02d}", "worker_supply": 12, "mineral": minerals,
                         "gas": 0, "supply_left": supply_left, "supply_cap": 23,
                         "supply_used": 23 - supply_left, "army_supply": 2 * (zealots + stalkers)}),
        "building": str({"nexus_count": 1, "pylon_count": 1, "gateway_count": 1}),
        "unit": str({"probe_count": 12, "Zealot_count": zealots, "stalker_count": stalkers}),
    }


def llm_rec(call_id, retries=0, error=None, final=False, hit=800, miss=200):
    return LLMCallRecord(
        call_id=call_id, game_step=0, game_time_s=0.0, purpose="L2_actor",
        model_requested="deepseek-v4-flash", model_returned=None if error else "deepseek-v4-flash-0501",
        reasoning_effort=None, prompt_tokens=0 if error else 1000, completion_tokens=0 if error else 50,
        reasoning_tokens=None, cache_hit_tokens=None if error else hit,
        cache_miss_tokens=None if error else miss, latency_s=1.5, retries=retries, error=error,
        finish_reason=None if error else "stop", output_chars=0 if error else 120,
        has_reasoning_content=False, final_failure=final)


def run_lifecycle(logger):
    logger.log_llm_call(llm_rec(0), messages=[{"role": "user", "content": f"key={SECRET}"}],
                        response_text="ACTIONS: 0, 19, 22, 23")
    logger.log_llm_call(llm_rec(1, error="Timeout"))
    logger.log_llm_call(llm_rec(1, retries=1))
    proposed, kept = [0, 19, 22, 64], [0, 19, EMPTY, 64]
    slots = mix_with_sources(list(kept), source_labels(proposed, kept, "structured", EMPTY), 10, 0.5, EMPTY)
    logger.log_tick(build_tick(0, 0, 0, 0.0, 5, "structured", proposed, kept, slots, EMPTY))
    for step, (aid, src) in enumerate(slots):
        logger.log_observation(step, float(step), raw_obs(step, supply_left=0 if step < 3 else 5,
                                                          zealots=2, stalkers=1),
                               action_failures=[],
                               extra={"scripted_attack_rule_steps": step, "scripted_attack_order_steps": max(0, step - 7),
                                      "scripted_attack_units_ordered": 4 * max(0, step - 7)})
        logger.log_slot_outcome(resolve_slot(aid, NAMES[aid], src, 0, step, float(step), NAMES[aid], []))
    logger.log_event("note", level="info", msg="hello")
    return logger.finish("Result.Victory", {})


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_disabled_lifecycle_writes_complete_jsonl_and_summary(tmp_path):
    logger = RunLogger(make_args(), str(tmp_path), mode="disabled", register_atexit=False)
    summary = run_lifecycle(logger)

    files = glob.glob(str(tmp_path / "*.jsonl"))
    assert {os.path.basename(f) for f in files} >= {
        "llm_calls.jsonl", "decision_ticks.jsonl", "slots.jsonl", "observations.jsonl", "events.jsonl"}
    for f in files:
        for line in read_jsonl(f):
            assert line["schema_version"] == SCHEMA_VERSION
            assert "type" in line and "ts" in line

    on_disk = json.load(open(tmp_path / SUMMARY_FILE))
    assert on_disk == summary
    for key in SUMMARY_KEYS:
        assert key in summary, key
    assert summary["result"] == "Victory"
    assert summary["n_llm_calls"] == 2
    assert summary["n_llm_errors"] == 1
    assert summary["total_retries"] == 1
    assert summary["model_resolved"] == "deepseek-v4-flash-0501"
    assert summary["cache_hit_ratio"] == pytest.approx(0.8)
    assert summary["est_cost_usd"] == pytest.approx((1600 * 0.1 + 400 * 1.0 + 100 * 2.0) / 1e6)
    assert summary["padding_share"] == pytest.approx(0.6)
    assert summary["nonreal_share"] == pytest.approx(0.7)
    assert summary["time_to_first_attack_s"] == 9.0      # MULTI-ATTACK is the last real slot
    assert summary["scripted_attack_rule_steps"] == 9
    assert summary["scripted_attack_order_steps"] == 2
    assert summary["scripted_attack_units_ordered"] == 8
    assert summary["scripted_attack_first_order_s"] == 8.0
    assert summary["supply_blocked_s"] == pytest.approx(3.0)
    assert summary["army_comp_entropy_mean"] > 0

    for path in glob.glob(str(tmp_path / "*")):
        assert SECRET not in open(path, encoding="utf-8").read(), path


def test_config_excludes_api_key(tmp_path):
    logger = RunLogger(make_args(), str(tmp_path), mode="disabled", register_atexit=False)
    assert "LLM_api_key" not in logger.config
    assert SECRET not in json.dumps(logger.config)
    assert logger.config["map"] == "Ancient Cistern LE"
    assert logger.run_name == "VeryEasy-AncientCistern-deepseek-v4-flash-20261023_120000-p-1"
    logger.finish("aborted")


@pytest.mark.parametrize("result,code,tag", [("Result.Victory", 0, None), ("aborted", 130, "aborted"),
                                             ("crash", 1, "crash")])
def test_abnormal_results_are_tagged_with_exit_code(tmp_path, result, code, tag):
    run = FakeRun()
    mod, _ = fake_wandb(run)
    logger = RunLogger(make_args(), str(tmp_path), mode="offline", wandb_module=mod, register_atexit=False)
    logger.finish(result)
    assert run.exit_code == code
    if tag:
        assert run.tags[-1] == tag and "phase0" in run.tags
    else:
        assert run.tags == ("deepseek", "phase0")


class FakeRun:
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.logged, self.metrics, self.artifacts = [], [], []
        self.summary = {}
        self.tags = ("deepseek", "phase0")
        self.config = types.SimpleNamespace(update=lambda *a, **k: None)
        self.exit_code = None

    def _maybe_fail(self, name):
        if name in self.fail_on:
            raise RuntimeError(f"wandb {name} exploded")

    def define_metric(self, *a, **k):
        self.metrics.append((a, k))

    def log(self, data):
        self._maybe_fail("log")
        self.logged.append(data)

    def log_artifact(self, art):
        self._maybe_fail("artifact")
        self.artifacts.append(art)

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class FakeArtifact:
    def __init__(self, name, type):
        self.name, self.type, self.files = name, type, []

    def add_file(self, path, name=None):
        self.files.append((path, name))


def fake_wandb(run):
    calls = {}

    def init(**kw):
        calls.update(kw)
        return run

    mod = types.SimpleNamespace(init=init, Artifact=FakeArtifact,
                                Table=lambda columns, data: {"columns": columns, "data": data})
    return mod, calls


def test_wandb_sink_receives_metrics_tables_and_artifact(tmp_path):
    (tmp_path / "raw_observation.json").write_text(json.dumps({"leak": SECRET}) + "\n")
    run = FakeRun()
    mod, init_kw = fake_wandb(run)
    logger = RunLogger(make_args(), str(tmp_path), mode="offline", wandb_module=mod, register_atexit=False)
    run_lifecycle(logger)

    assert init_kw["project"] == "textsc2"
    assert init_kw["group"] == "unit"
    assert init_kw["job_type"] == "game"
    assert init_kw["mode"] == "offline"
    assert "phase0" in init_kw["tags"]
    assert "LLM_api_key" not in init_kw["config"]
    assert any(a == ("game/*",) and k == {"step_metric": "game_time_s"} for a, k in run.metrics)
    keys = set().union(*(d.keys() for d in run.logged))
    assert {"funnel/parse_rate", "llm/latency_s", "game/minerals", "game/army_comp_entropy",
            "decision_ticks", "failures", "action_histogram"} <= keys
    assert run.summary["result"] == "Victory"
    assert run.exit_code == 0

    art = run.artifacts[0]
    assert art.type == "game_logs" and art.name.startswith("game-")
    names = {n for _, n in art.files}
    assert {"llm_calls.jsonl", "game_summary.json", "raw_observation.json"} <= names
    for path, _ in art.files:
        assert SECRET not in open(path, encoding="utf-8").read(), path


def test_wandb_failure_degrades_to_jsonl_only(tmp_path, capsys):
    run = FakeRun(fail_on={"log"})
    mod, _ = fake_wandb(run)
    logger = RunLogger(make_args(), str(tmp_path), mode="online", wandb_module=mod, register_atexit=False)
    with pytest.warns(UserWarning):
        summary = run_lifecycle(logger)
    assert not logger.wandb_active
    assert capsys.readouterr().out.count("wandb error") == 1
    assert len(read_jsonl(tmp_path / "observations.jsonl")) == 10
    assert summary["result"] == "Victory"
    events = [e["type"] for e in read_jsonl(tmp_path / "events.jsonl")]
    assert "wandb_disabled" in events


def test_wandb_init_failure_does_not_raise(tmp_path):
    def boom(**kw):
        raise RuntimeError("not logged in")
    mod = types.SimpleNamespace(init=boom)
    with pytest.warns(UserWarning):
        logger = RunLogger(make_args(), str(tmp_path), mode="online", wandb_module=mod, register_atexit=False)
    assert not logger.wandb_active
    assert run_lifecycle(logger)["result"] == "Victory"


def test_mode_resolution(monkeypatch, tmp_path):
    from sc2_rl_agent.starcraftenv_test.telemetry.run_logger import resolve_mode
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert resolve_mode(None) == "offline"
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert resolve_mode(None) == "disabled"
    assert resolve_mode("online") == "online"


# ---------------------------------------------------------------------------
# LLM client: behaviour unchanged, every attempt metered
# ---------------------------------------------------------------------------

def _response(content="ACTIONS: 0, 19", model="deepseek-v4-flash-0501"):
    usage = types.SimpleNamespace(prompt_tokens=900, completion_tokens=40, completion_tokens_details=None,
                                  prompt_cache_hit_tokens=700, prompt_cache_miss_tokens=200)
    message = types.SimpleNamespace(content=content, reasoning_content=None)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=usage, model=model)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kw):
        self.requests.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make_bot(outcomes, monkeypatch):
    from sc2_rl_agent.starcraftenv_test.LLM import gpt_test
    monkeypatch.setattr(gpt_test.time, "sleep", lambda s: None)
    bot = gpt_test.ChatBot_SingleTurn("deepseek-v4-flash", SECRET, "http://127.0.0.1:9", "sys",
                                      ["example in", "example out"], 0.3)
    completions = FakeCompletions(outcomes)
    bot.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    records = []
    bot.meter = LLMMeter(sink=lambda rec, **kw: records.append((rec, kw)), model_requested="deepseek-v4-flash")
    return bot, completions, records


def test_chatbot_retries_and_meters_each_attempt(monkeypatch):
    bot, completions, records = make_bot([RuntimeError("502"), _response()], monkeypatch)
    assert bot.query("state") == "ACTIONS: 0, 19"
    req = completions.requests[0]
    assert req["model"] == "deepseek-v4-flash" and req["temperature"] == 0.3
    assert [m["role"] for m in req["messages"]] == ["system", "user", "assistant", "user"]
    assert "extra_body" not in req
    assert [r.retries for r, _ in records] == [0, 1]
    assert records[0][0].error.startswith("RuntimeError")
    ok = records[1][0]
    assert ok.model_returned == "deepseek-v4-flash-0501"
    assert (ok.cache_hit_tokens, ok.cache_miss_tokens) == (700, 200)
    assert ok.call_id == records[0][0].call_id
    assert records[1][1]["messages"][-1]["content"] == "state"


def test_chatbot_all_retries_fail_returns_original_apology(monkeypatch):
    bot, _, records = make_bot([RuntimeError("down")] * 5, monkeypatch)
    out = bot.query("state")
    assert out == ("I'm sorry, but I am unable to provide a response at this time due to "
                   "technical difficulties.")
    assert len(records) == 5 and records[-1][0].final_failure
    assert bot.meter.pop_completed()[0]["failed"] is True


def test_chatbot_reasoning_effort_and_meter_errors(monkeypatch):
    bot, completions, _ = make_bot([_response()], monkeypatch)
    bot.reasoning_effort = "high"
    bot.meter = types.SimpleNamespace(begin=lambda m: (_ for _ in ()).throw(RuntimeError("meter broke")))
    assert bot.query("state") == "ACTIONS: 0, 19"
    assert completions.requests[0]["extra_body"] == {"reasoning_effort": "high"}


# ---------------------------------------------------------------------------
# Agent hooks: a tiny simulated game through AgentTelemetry
# ---------------------------------------------------------------------------

def test_agent_telemetry_simulated_game(tmp_path):
    args = make_args(wandb_mode="disabled", prompt_K=5)
    chatbot = types.SimpleNamespace(meter=None)
    agent = types.SimpleNamespace(game_folder=str(tmp_path), model_name="deepseek-v4-flash",
                                  L2=types.SimpleNamespace(chatbot=chatbot), empty_action_idx=EMPTY,
                                  _id_to_name=NAMES)
    tm = AgentTelemetry.create(agent, args, extra_config={"prompt_type": "v2"})
    assert tm.enabled and chatbot.meter is tm.meter

    queue = []
    executed, failures = [], []
    for step in range(12):
        obs = {"information": raw_obs(step), "action_executed": executed[-1] if executed else [],
               "action_failures": failures[-1] if failures else [], "game_loop": int(step * 22.4),
               "scripted_attack_rule_steps": 0}
        tm.on_observation(obs)
        if step % 10 == 0:
            tm.before_llm()
            call = tm.meter.begin([{"role": "user", "content": "x"}])
            call.success(0, tm.meter.clock(), _response(), "ACTIONS: 0, 19")
            proposed, kept = [0, 19], [0, 19]
            slots = mix_with_sources(list(kept), source_labels(proposed, kept, "structured", EMPTY),
                                     10, 0.5, EMPTY)
            tick_id = tm.on_tick("ACTIONS: 0, 19", "structured", proposed, kept, slots)
            queue.extend((a, s, tick_id) for a, s in slots)
        aid, src, tid = queue.pop(0) if queue else (EMPTY, "queue_empty", None)
        tm.on_dequeue(aid, src, tid)
        executed.append(NAMES[aid])
        failures.append(["Action failed: BUILD PYLON, Reason: placement failed"] if aid == 19 else [])

    final = {"action_executed": executed[-1], "action_failures": failures[-1],
             "replay_path": "/tmp/x.SC2Replay", "scripted_attack_rule_steps": 3,
             "scripted_attack_order_steps": 1, "scripted_attack_units_ordered": 6}
    summary = tm.finish("Result.Defeat", final)
    assert summary["result"] == "Defeat"
    assert summary["n_steps"] == 12 and summary["n_ticks"] == 2
    assert summary["llm_slots_dequeued"] == 3          # 2 from tick 0, 1 from tick 1 (step 10)
    assert summary["exec_success"] == pytest.approx(2 / 3)   # the BUILD PYLON slot failed
    assert summary["scripted_attack_rule_steps"] == 3
    assert summary["scripted_attack_order_steps"] == 1
    assert summary["scripted_attack_units_ordered"] == 6
    assert summary["replay_path"] == "/tmp/x.SC2Replay"
    slots = read_jsonl(tmp_path / "slots.jsonl")
    assert len(slots) == 12 and all(s["resolved"] for s in slots)
    assert slots[0]["resolve_step"] == 1
    ticks = read_jsonl(tmp_path / "decision_ticks.jsonl")
    assert ticks[0]["call_id"] == 0 and ticks[1]["call_id"] == 1
