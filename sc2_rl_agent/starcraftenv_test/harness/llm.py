"""
Structured decisions from any OpenAI-compatible server.

The decision schema is fixed for the whole run (ids are any action id, the
per-step legal subset is given in the user message and enforced afterwards),
so tool definitions never change and the prompt prefix stays cacheable.

Transport ladder, strongest first, chosen per server and remembered:
  tools        forced function call; arguments follow the schema (DeepSeek, vLLM, OpenAI)
  json_schema  constrained decoding (vLLM / OpenAI; DeepSeek rejects it)
  json_object  JSON mode, schema given in the prompt
  text         plain completion, first JSON object extracted from the text
A server that rejects a mode with HTTP 400 is downgraded once, not retried.
"""

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MODES = ("tools", "json_schema", "json_object", "text")
TOOL_NAME = "submit_decision"


def decision_schema(max_actions, n_action_ids):
    return {
        "type": "object",
        "properties": {
            "plan": {"type": "string",
                     "description": "Standing strategy in 1-2 short sentences; it is shown to you next turn."},
            "actions": {
                "type": "array",
                "maxItems": max_actions,
                "description": "Actions to queue now, most important first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "minimum": 0, "maximum": n_action_ids - 1},
                        "why": {"type": "string", "description": "A few words."},
                    },
                    "required": ["id", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plan", "actions"],
        "additionalProperties": False,
    }


@dataclass
class Decision:
    plan: str = ""
    actions: List[Tuple[int, str]] = field(default_factory=list)   # (id, why)

    @property
    def action_ids(self):
        return [a for a, _ in self.actions]


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text):
    if not isinstance(text, str):
        return None
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for c in candidates:
        try:
            return json.loads(c)
        except ValueError:
            continue
    return None


def parse_decision(payload, max_actions, n_action_ids) -> Tuple[Optional[Decision], List[str]]:
    """Validate a decision. Returns (decision or None, problems)."""
    obj = payload if isinstance(payload, dict) else _extract_json(payload)
    if not isinstance(obj, dict):
        return None, ["reply is not a JSON object"]
    problems = []
    plan = obj.get("plan", "")
    if not isinstance(plan, str):
        problems.append("'plan' must be a string")
        plan = str(plan)
    raw = obj.get("actions")
    if not isinstance(raw, list):
        return None, problems + ["'actions' must be a list"]
    actions = []
    for item in raw:
        aid, why = (item.get("id"), item.get("why", "")) if isinstance(item, dict) else (item, "")
        if isinstance(aid, str) and aid.strip().isdigit():
            aid = int(aid)
        if not isinstance(aid, int) or isinstance(aid, bool) or not 0 <= aid < n_action_ids:
            problems.append(f"invalid action id {aid!r}")
            continue
        actions.append((aid, str(why)[:200]))
    if len(actions) > max_actions:
        problems.append(f"too many actions ({len(actions)} > {max_actions}); extra ones ignored")
        actions = actions[:max_actions]
    return Decision(plan=plan.strip()[:500], actions=actions), problems


def thinking_extra_body(thinking, provider, base_url):
    """How to switch the model's hidden reasoning on/off for this server."""
    if thinking == "auto":
        return {}
    on = thinking == "enabled"
    if provider == "auto":
        provider = "deepseek" if "deepseek" in (base_url or "") else "chat_template"
    if provider == "deepseek":
        return {"thinking": {"type": "enabled" if on else "disabled"}}
    if provider == "chat_template":          # vLLM / SGLang serving Qwen3-style models
        return {"chat_template_kwargs": {"enable_thinking": on}}
    return {}


def _is_unsupported(exc):
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status in (400, 404, 422) and any(w in text for w in (
        "response_format", "unavailable", "not support", "unsupported", "tool", "json_schema", "guided"))


class StructuredClient:
    """Chat client that returns validated Decisions. Mirrors ChatBot_SingleTurn's
    transport behaviour (5 tries, exponential backoff, 1800 s timeout)."""

    def __init__(self, model, api_key, base_url, temperature, max_actions, n_action_ids,
                 mode="auto", thinking="disabled", thinking_provider="auto", max_tokens=1024,
                 max_retries=5, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=1800, max_retries=0)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_actions = max_actions
        self.n_action_ids = n_action_ids
        self.schema = decision_schema(max_actions, n_action_ids)
        self.modes = list(MODES) if mode == "auto" else [mode]
        self.extra_body = thinking_extra_body(thinking, thinking_provider, base_url)
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.meter = None          # telemetry LLMMeter, attached by AgentTelemetry
        self.reasoning_effort = None

    @property
    def mode(self):
        return self.modes[0]

    # ------------------------------------------------------------------

    def _request(self, messages, mode):
        req = dict(model=self.model, messages=messages, temperature=self.temperature,
                   max_tokens=self.max_tokens)
        extra = dict(self.extra_body)
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort
        if extra:
            req["extra_body"] = extra
        if mode == "tools":
            req["tools"] = [{"type": "function", "function": {
                "name": TOOL_NAME, "description": "Submit this turn's decision.",
                "parameters": self.schema, "strict": True}}]
            req["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
        elif mode == "json_schema":
            req["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "decision", "schema": self.schema, "strict": True}}
        elif mode == "json_object":
            req["response_format"] = {"type": "json_object"}
        return req

    @staticmethod
    def _payload(response, mode):
        msg = response.choices[0].message
        calls = getattr(msg, "tool_calls", None)
        if calls:
            return calls[0].function.arguments
        return msg.content

    def _meter(self, call, method, *args):
        if call is None:
            return
        try:
            getattr(call, method)(*args)
        except Exception as e:
            print(f"[telemetry] LLM meter error ignored: {e}")

    def _call(self, messages):
        """One logical call with transport retries and mode downgrade.
        Returns (payload text or None, mode used)."""
        call = None
        if self.meter is not None:
            try:
                call = self.meter.begin(list(messages))
            except Exception as e:
                print(f"[telemetry] LLM meter error ignored: {e}")
        attempt = 0
        while True:
            mode = self.mode
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**self._request(messages, mode))
                payload = self._payload(resp, mode)
                self._meter(call, "success", attempt, t0, resp, payload)
                return payload, mode
            except Exception as e:
                if _is_unsupported(e) and len(self.modes) > 1:
                    print(f"[harness] server rejected structured mode '{mode}', downgrading: {str(e)[:160]}")
                    self.modes.pop(0)
                    self._meter(call, "error", attempt, t0, e, False)
                    attempt += 1
                    continue
                print(f"Error when calling the LLM API: {e}")
                final = attempt >= self.max_retries - 1
                self._meter(call, "error", attempt, t0, e, final)
                if final:
                    print("Maximum number of retries reached. The LLM API is not responding.")
                    return None, mode
                sleep_time = (2 ** attempt) + random.random()
                print(f"Waiting for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
                attempt += 1

    def decide(self, messages) -> Tuple[Optional[Decision], str, Optional[str], List[str]]:
        """Returns (decision, parse_tier, raw payload, problems).
        parse_tier is the transport mode, 'repair' after a corrective retry,
        or 'none' when no valid decision was obtained."""
        payload, mode = self._call(messages)
        if payload is None:
            return None, "none", None, ["LLM call failed"]
        decision, problems = parse_decision(payload, self.max_actions, self.n_action_ids)
        if decision is not None and (decision.actions or not problems):
            return decision, mode if mode != "text" else "json_text", payload, problems
        # One corrective retry with the validator's feedback.
        repair = list(messages) + [
            {"role": "assistant", "content": payload if isinstance(payload, str) else json.dumps(payload)},
            {"role": "user", "content": "That reply was not a valid decision: " + "; ".join(problems)
             + ". Reply again with only the decision object."},
        ]
        payload2, _ = self._call(repair)
        decision2, problems2 = parse_decision(payload2, self.max_actions, self.n_action_ids) \
            if payload2 is not None else (None, ["LLM call failed"])
        if decision2 is not None and (decision2.actions or not problems2):
            return decision2, "repair", payload2, problems + problems2
        return None, "none", payload2 if payload2 is not None else payload, problems + problems2
