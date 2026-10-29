"""
Minimal OpenAI-compatible chat server for smoke-testing a full game without
spending tokens or a GPU (same API shape as vLLM / Ollama / DeepSeek).

    python -m sc2_rl_agent.starcraftenv_test.tests.mock_llm_server --port 8000
    python -m sc2_rl_agent.starcraftenv_test.test_the_env --agent_type deepseek \
        --LLM_api_base http://127.0.0.1:8000/v1 --LLM_model_name mock --LLM_api_key x ...

Plain requests get text ending in "ACTIONS: a, b, c, d, e" (K random valid
ids) for the regex extractor. Requests with `tools` get a forced tool call and
requests with `response_format` get JSON content, both shaped like the
harness decision {"plan", "actions": [{"id", "why"}]}; ids are drawn from the
"Legal action ids" line of the last message when present. --illegal_rate
mixes in an id outside that list, --fail_rate returns HTTP 500 sometimes, and
--reject_tools answers tool requests with HTTP 400 to test mode downgrade.
"""

import argparse
import json
import random
import threading
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sc2_rl_agent.starcraftenv_test.utils.action_info import ActionDescriptions


_LEGAL = re.compile(r"Legal action ids[^\n]*\n([0-9, ]+)")


def make_handler(action_ids, k, model, fail_rate, rng, illegal_rate=0.0, reject_tools=False):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            messages = request.get("messages", [])
            if reject_tools and request.get("tools"):
                self._send(400, {"error": {"message": "tool use is not supported", "type": "invalid_request_error"}})
                return
            m = _LEGAL.search(str(messages[-1].get("content", ""))) if messages else None
            pool = [int(x) for x in m.group(1).replace(" ", "").split(",") if x] if m else action_ids
            with lock:
                fail = rng.random() < fail_rate
                picks = rng.sample(pool, min(k, len(pool)))
                if rng.random() < illegal_rate:
                    outside = [a for a in action_ids if a not in pool]
                    if outside:
                        picks[-1:] = [rng.choice(outside)]
            if fail:
                self._send(500, {"error": {"message": "mock failure", "type": "server_error"}})
                return
            prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
            prompt_tokens = max(1, prompt_chars // 4)
            decision = json.dumps({"plan": "mock plan: macro up",
                                   "army_goal": {"mode": "attack", "units": [{"unit": "zealot", "count": 6}],
                                                 "attack_at_supply": 60},
                                   "actions": [{"id": a, "why": "mock"} for a in picks]})
            if request.get("tools"):
                tools = {t["function"]["name"]: t["function"] for t in request["tools"]}
                choice = request.get("tool_choice")
                forced = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
                user = str(messages[-1].get("content", "")) if messages else ""
                if forced is None and "load_skill" in tools and "## Active skill\n(none loaded)" in user:
                    skills = tools["load_skill"]["parameters"]["properties"]["name"]["enum"]
                    name, arguments = "load_skill", json.dumps({"name": skills[0]})
                else:
                    name = forced or next(n for n in tools if n != "load_skill")
                    arguments = decision
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call_{time.time_ns()}", "type": "function",
                    "function": {"name": name, "arguments": arguments}}]}
                finish = "tool_calls"
            elif request.get("response_format"):
                message, finish = {"role": "assistant", "content": decision}, "stop"
            else:
                message = {"role": "assistant",
                           "content": "Mock decision.\nACTIONS: " + ", ".join(str(a) for a in picks)}
                finish = "stop"
            self._send(200, {
                "id": f"mock-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 20,
                          "total_tokens": prompt_tokens + 20},
            })

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--race", default="Protoss")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default="mock-model")
    p.add_argument("--fail_rate", type=float, default=0.0)
    p.add_argument("--illegal_rate", type=float, default=0.0)
    p.add_argument("--reject_tools", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    desc = ActionDescriptions(a.race)
    action_ids = [i for i in desc.flattened_actions if i != desc.empty_action_id]
    server = ThreadingHTTPServer((a.host, a.port),
                                 make_handler(action_ids, a.k, a.model, a.fail_rate, random.Random(a.seed),
                                              a.illegal_rate, a.reject_tools))
    print(f"mock LLM server on http://{a.host}:{a.port}/v1 (model={a.model})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
