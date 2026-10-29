import json
import os
import socket
import threading
from argparse import Namespace

import pytest

from sc2_rl_agent.starcraftenv_test import sc2_runtime


@pytest.fixture
def fake_sc2(tmp_path):
    root = tmp_path / "StarCraftII"
    (root / "Versions" / "Base75689").mkdir(parents=True)
    (root / "Versions" / "Base75689" / "SC2_x64").write_text("")
    (root / "Maps" / "Ladder").mkdir(parents=True)
    (root / "Maps" / "Ladder" / "Ancient Cistern LE.SC2Map").write_text("")
    return root


def test_apply_args_exports_platform_and_path(monkeypatch, fake_sc2):
    monkeypatch.delenv("SC2_WSL_DETECT", raising=False)
    monkeypatch.setenv("SC2CLIENTHOST", "172.26.144.1")
    monkeypatch.setenv("SC2SERVERHOST", "0.0.0.0")
    monkeypatch.setenv("SC2PF", "WSL2")
    monkeypatch.setenv("SC2PATH", "/mnt/c/Program Files (x86)/StarCraft II/")
    info = sc2_runtime.apply_sc2_runtime_args(
        ["--agent_type", "deepseek", "--sc2_platform", "Linux", "--sc2_path", str(fake_sc2)])
    assert os.environ["SC2PF"] == "Linux"
    assert os.environ["SC2PATH"] == str(fake_sc2)
    # Without this burnysc2 rewrites Linux -> WSL2 inside WSL and launches Windows SC2.
    assert os.environ["SC2_WSL_DETECT"] == "0"
    assert os.environ["SC2CLIENTHOST"] == os.environ["SC2SERVERHOST"] == "127.0.0.1"
    assert info == {"sc2_platform": "Linux", "sc2_path": str(fake_sc2),
                    "sc2_base_build": "75689", "sc2_headless": True}


def test_without_flags_shell_env_is_kept(monkeypatch):
    monkeypatch.setenv("SC2PF", "WSL2")
    monkeypatch.setenv("SC2PATH", "/some/where")
    info = sc2_runtime.apply_sc2_runtime_args(["--difficulty", "Hard"])
    assert info["sc2_platform"] == "WSL2" and info["sc2_headless"] is False
    assert os.environ["SC2PATH"] == "/some/where"


def test_check_install_finds_nested_map(monkeypatch, fake_sc2):
    monkeypatch.setenv("SC2PF", "Linux")
    monkeypatch.setenv("SC2PATH", str(fake_sc2))
    assert sc2_runtime.check_install("Ancient Cistern LE")["sc2_base_build"] == "75689"


def test_check_install_reports_missing_map_and_binary(monkeypatch, tmp_path):
    root = tmp_path / "empty"
    (root / "Maps").mkdir(parents=True)
    monkeypatch.setenv("SC2PF", "Linux")
    monkeypatch.setenv("SC2PATH", str(root))
    with pytest.raises(SystemExit) as e:
        sc2_runtime.check_install("Altitude LE")
    msg = str(e.value)
    assert "SC2_x64" in msg and "Altitude LE" in msg


def test_run_config_records_sc2_runtime_and_running_version(monkeypatch, tmp_path, fake_sc2):
    from sc2_rl_agent.starcraftenv_test.telemetry.run_logger import RunLogger
    monkeypatch.setenv("SC2PF", "Linux")
    monkeypatch.setenv("SC2PATH", str(fake_sc2))
    logger = RunLogger(Namespace(LLM_api_key="x", difficulty="VeryEasy"), str(tmp_path),
                       mode="disabled", register_atexit=False)
    assert logger.config["sc2_platform"] == "Linux"
    assert logger.config["sc2_base_build"] == "75689"
    assert logger.config["sc2_version"] is None
    logger.log_observation(0, 0.0, {}, extra={"sc2_game_version": "4.10.0.75689"})
    assert logger.config["sc2_version"] == "4.10.0.75689"
    logger.finish("Result.Tie")


def test_short_local_api_key_is_not_scrubbed(tmp_path):
    from sc2_rl_agent.starcraftenv_test.telemetry.run_logger import RunLogger
    logger = RunLogger(Namespace(LLM_api_key="x", LLM_model_name="qwen-max"), str(tmp_path),
                       mode="disabled", register_atexit=False)
    logger.log_event("note", text="max next")
    logger.finish("Result.Tie")
    assert "max next" in open(tmp_path / "events.jsonl").read()
    assert json.load(open(tmp_path / "game_summary.json"))["model_requested"] == "qwen-max"


# ---------------------------------------------------------------------------
# Real HTTP: migrated openai>=1 client -> mock server -> extractor
# ---------------------------------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_server():
    from http.server import ThreadingHTTPServer
    import random
    from sc2_rl_agent.starcraftenv_test.tests.mock_llm_server import make_handler
    servers = []

    def start(fail_rate=0.0):
        port = _free_port()
        srv = ThreadingHTTPServer(("127.0.0.1", port),
                                  make_handler([0, 19, 22, 23, 1, 2, 3], 5, "mock-model", fail_rate,
                                               random.Random(1)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{port}/v1"

    yield start
    for srv in servers:
        srv.shutdown()


def test_chatbot_against_mock_server_end_to_end(mock_server):
    from sc2_rl_agent.starcraftenv_test.LLM.gpt_test import ChatBot_SingleTurn
    from sc2_rl_agent.starcraftenv_test.telemetry.llm_meter import LLMMeter
    from sc2_rl_agent.starcraftenv_test.utils.action_extractor import ActionExtractor, extract_actions_from_command
    from sc2_rl_agent.starcraftenv_test.utils.action_info import ActionDescriptions

    bot = ChatBot_SingleTurn("mock-model", "x", mock_server(), "sys", ["in", "out"], 0.3)
    records = []
    bot.meter = LLMMeter(sink=lambda rec, **kw: records.append(rec), model_requested="mock-model")
    answer = bot.query("state")
    assert "ACTIONS:" in answer

    desc = ActionDescriptions("Protoss")
    info = {}
    ids, _ = extract_actions_from_command(answer, ActionExtractor(desc.action_descriptions),
                                          desc.empty_action_id, None, info=info)
    assert info["parse_tier"] == "structured" and len(ids) == 5
    rec = records[0]
    assert rec.error is None and rec.model_returned == "mock-model" and rec.prompt_tokens > 0
    # A local server has no DeepSeek cache fields; they are recorded as None.
    assert rec.cache_hit_tokens is None and rec.cache_miss_tokens is None


def test_chatbot_retries_against_failing_server(mock_server, monkeypatch):
    from sc2_rl_agent.starcraftenv_test.LLM import gpt_test
    monkeypatch.setattr(gpt_test.time, "sleep", lambda s: None)
    bot = gpt_test.ChatBot_SingleTurn("mock-model", "x", mock_server(fail_rate=1.0), "sys", ["in", "out"], 0.3)
    assert bot.query("state", max_retries=3).startswith("I'm sorry")
