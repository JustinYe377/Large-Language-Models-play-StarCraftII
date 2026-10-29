"""
Minimal OpenAI-compatible chat server for smoke-testing a full game without
spending tokens or a GPU (same API shape as vLLM / Ollama / DeepSeek).

    python -m sc2_rl_agent.starcraftenv_test.tests.mock_llm_server --port 8000
    python -m sc2_rl_agent.starcraftenv_test.test_the_env --agent_type deepseek \
        --LLM_api_base http://127.0.0.1:8000/v1 --LLM_model_name mock --LLM_api_key x ...

Every reply ends with "ACTIONS: a, b, c, d, e" (K random valid action ids), so
the structured extractor tier parses it. --fail_rate makes some requests
return HTTP 500 to exercise the client's retry path.
"""

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sc2_rl_agent.starcraftenv_test.utils.action_info import ActionDescriptions


def make_handler(action_ids, k, model, fail_rate, rng):
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
            with lock:
                fail = rng.random() < fail_rate
                picks = rng.sample(action_ids, k)
            if fail:
                self._send(500, {"error": {"message": "mock failure", "type": "server_error"}})
                return
            prompt_chars = sum(len(str(m.get("content", ""))) for m in request.get("messages", []))
            prompt_tokens = max(1, prompt_chars // 4)
            content = "Mock decision.\nACTIONS: " + ", ".join(str(a) for a in picks)
            self._send(200, {
                "id": f"mock-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
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
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    desc = ActionDescriptions(a.race)
    action_ids = [i for i in desc.flattened_actions if i != desc.empty_action_id]
    server = ThreadingHTTPServer((a.host, a.port),
                                 make_handler(action_ids, a.k, a.model, a.fail_rate, random.Random(a.seed)))
    print(f"mock LLM server on http://{a.host}:{a.port}/v1 (model={a.model})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
