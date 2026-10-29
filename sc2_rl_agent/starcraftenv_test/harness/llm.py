"""
Structured decisions from any OpenAI-compatible server.

The decision schema and tool definitions are fixed for the whole run (ids are
any action id; the per-step legal subset is given in the user message and
enforced afterwards), so they never change and the prompt prefix stays
cacheable.

Transport ladder, strongest first, chosen per server and remembered:
  tools        function calls: submit_decision (+ load_skill when skills exist)
  json_schema  constrained decoding (vLLM / OpenAI; DeepSeek rejects it)
  json_object  JSON mode, schema given in the prompt
  text         plain completion, first JSON object extracted from the text
A server that rejects a mode with HTTP 400 is downgraded once, not retried.

Skills (progressive disclosure): in tools mode the model may call
load_skill(name) first; the skill text is returned as the tool result and a
second, forced submit_decision call follows. In JSON modes the decision's
"load_skill" field does the same. At most one skill round per decision.
"""

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MODES = ("tools", "json_schema", "json_object", "text")
TOOL_NAME = "submit_decision"
SKILL_TOOL = "load_skill"
GOAL_MODES = ("keep", "attack", "defend")
ARMY_UNITS = ("zealot", "stalker", "sentry", "adept", "high_templar", "dark_templar", "immortal",
              "colossus", "disruptor", "archon", "observer", "warp_prism", "phoenix", "voidray",
              "oracle", "carrier", "tempest", "mothership")


def decision_schema(max_actions, n_action_ids, with_load_skill=False):
    props = {
        "plan": {"type": "string",
                 "description": "Standing strategy in 1-2 short sentences; it is shown to you next turn."},
        "army_goal": {
            "type": "object",
            "description": 'The army you are building. mode "keep" leaves the current goal unchanged.',
            "properties": {
                "mode": {"type": "string", "enum": list(GOAL_MODES)},
                "units": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"unit": {"type": "string", "enum": list(ARMY_UNITS)},
                                   "count": {"type": "integer", "minimum": 0, "maximum": 60}},
                    "required": ["unit", "count"], "additionalProperties": False}},
                "attack_at_supply": {"type": "integer", "minimum": 0, "maximum": 200},
            },
            "required": ["mode", "units", "attack_at_supply"],
            "additionalProperties": False,
        },
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
    }
    required = ["plan", "army_goal", "actions"]
    if with_load_skill:
        props["load_skill"] = {"type": "string",
                               "description": 'Name of a strategy skill to read before deciding, or "".'}
        required.append("load_skill")
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def skill_tool(skill_names):
    return {"type": "function", "function": {
        "name": SKILL_TOOL,
        "description": "Read a strategy skill (build order, target army, when to attack) before deciding.",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string", "enum": list(skill_names)}},
                       "required": ["name"], "additionalProperties": False},
        "strict": True}}


@dataclass
class Decision:
    plan: str = ""
    actions: List[Tuple[int, str]] = field(default_factory=list)   # (id, why)
    army_goal: Optional[dict] = None     # {"mode", "units": {unit: n}, "attack_at_supply"}; None = keep
    load_skill: Optional[str] = None

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


def _parse_goal(raw, problems):
    if not isinstance(raw, dict):
        return None
    mode = raw.get("mode", "keep")
    if mode not in GOAL_MODES:
        problems.append(f"army_goal.mode must be one of {GOAL_MODES}")
        return None
    if mode == "keep":
        return None
    units = {}
    items = raw.get("units", [])
    if isinstance(items, dict):                       # tolerate {"stalker": 8}
        items = [{"unit": k, "count": v} for k, v in items.items()]
    for item in items if isinstance(items, list) else []:
        unit, count = (item.get("unit"), item.get("count")) if isinstance(item, dict) else (None, None)
        if unit in ARMY_UNITS and isinstance(count, int) and count > 0:
            units[unit] = min(count, 60)
        elif unit is not None:
            problems.append(f"army_goal unit {unit!r} ignored")
    supply = raw.get("attack_at_supply", 0)
    return {"mode": mode, "units": units, "attack_at_supply": int(supply) if isinstance(supply, int) else 0}


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
    skill = obj.get("load_skill")
    return Decision(plan=plan.strip()[:500], actions=actions, army_goal=_parse_goal(obj.get("army_goal"), problems),
                    load_skill=skill if isinstance(skill, str) and skill else None), problems


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


@dataclass
class Reply:
    """One HTTP response: tool calls [(id, name, arguments str)] and/or content."""
    calls: List[Tuple[str, str, str]]
    content: Optional[str]

    def payload_for(self, name):
        for _, n, args in self.calls:
            if n == name:
                return args
        return None

    def as_text(self):
        if self.calls:
            return json.dumps([{"name": n, "arguments": a} for _, n, a in self.calls])
        return self.content


class StructuredClient:
    """Chat client that returns validated Decisions. Mirrors ChatBot_SingleTurn's
    transport behaviour (5 tries, exponential backoff, 1800 s timeout)."""

    def __init__(self, model, api_key, base_url, temperature, max_actions, n_action_ids,
                 mode="auto", thinking="disabled", thinking_provider="auto", max_tokens=1024,
                 max_retries=5, client=None, skills=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=1800, max_retries=0)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_actions = max_actions
        self.n_action_ids = n_action_ids
        self.skills = skills                       # SkillLibrary or None
        self.skill_names = list(skills.names) if skills else []
        self.schema = decision_schema(max_actions, n_action_ids)
        self.json_schema = decision_schema(max_actions, n_action_ids, with_load_skill=bool(self.skill_names))
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

    def _tools(self):
        tools = [{"type": "function", "function": {
            "name": TOOL_NAME, "description": "Submit this turn's decision.",
            "parameters": self.schema, "strict": True}}]
        if self.skill_names:
            tools.append(skill_tool(self.skill_names))
        return tools

    def _request(self, messages, mode, allow_skill):
        req = dict(model=self.model, messages=messages, temperature=self.temperature,
                   max_tokens=self.max_tokens)
        extra = dict(self.extra_body)
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort
        if extra:
            req["extra_body"] = extra
        if mode == "tools":
            # Same tool list every call (cache); allow_skill only changes tool_choice.
            req["tools"] = self._tools()
            req["tool_choice"] = ("required" if allow_skill and self.skill_names
                                  else {"type": "function", "function": {"name": TOOL_NAME}})
        elif mode == "json_schema":
            req["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "decision", "schema": self.json_schema, "strict": True}}
        elif mode == "json_object":
            req["response_format"] = {"type": "json_object"}
        return req

    @staticmethod
    def _reply(response):
        msg = response.choices[0].message
        calls = []
        for i, c in enumerate(getattr(msg, "tool_calls", None) or []):
            calls.append((getattr(c, "id", None) or f"call_{i}", c.function.name, c.function.arguments))
        return Reply(calls, msg.content)

    def _meter(self, call, method, *args):
        if call is None:
            return
        try:
            getattr(call, method)(*args)
        except Exception as e:
            print(f"[telemetry] LLM meter error ignored: {e}")

    def _call(self, messages, allow_skill=False):
        """One logical call with transport retries and mode downgrade.
        Returns (Reply or None, mode used)."""
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
                resp = self.client.chat.completions.create(**self._request(messages, mode, allow_skill))
                reply = self._reply(resp)
                self._meter(call, "success", attempt, t0, resp, reply.as_text())
                return reply, mode
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

    def _skill_text(self, name):
        skill = self.skills.get(name) if self.skills else None
        if skill is None:
            return None
        return f"Skill {skill.name} ({skill.phase})\n\n{skill.body}"

    def decide(self, messages, allow_skill=True):
        """Returns (decision, parse_tier, raw payload text, problems, loaded skill names).
        parse_tier is the transport mode, 'repair' after a corrective retry, or
        'none' when no valid decision was obtained."""
        msgs = list(messages)
        loaded, problems = [], []
        skill_round = bool(allow_skill and self.skill_names)
        reply, mode = self._call(msgs, allow_skill=skill_round)

        for _ in range(2):                       # at most one skill round
            if reply is None:
                return None, "none", None, problems + ["LLM call failed"], loaded
            if mode == "tools":
                skill_calls = [(cid, args) for cid, n, args in reply.calls if n == SKILL_TOOL]
                if skill_calls and skill_round:
                    tool_msgs = []
                    for cid, args in skill_calls:
                        try:
                            name = json.loads(args).get("name")
                        except (ValueError, AttributeError):
                            name = None
                        text = self._skill_text(name)
                        if text:
                            loaded.append(name)
                        tool_msgs.append({"role": "tool", "tool_call_id": cid,
                                          "content": text or f"Unknown skill {name!r}."})
                    if reply.payload_for(TOOL_NAME) is None:
                        msgs = msgs + [{"role": "assistant", "content": reply.content or "",
                                        "tool_calls": [{"id": cid, "type": "function",
                                                        "function": {"name": n, "arguments": a}}
                                                       for cid, n, a in reply.calls]}] + tool_msgs
                        skill_round = False
                        reply, mode = self._call(msgs, allow_skill=False)
                        continue
                payload = reply.payload_for(TOOL_NAME) or reply.content
            else:
                payload = reply.content
            decision, p = parse_decision(payload, self.max_actions, self.n_action_ids)
            problems += p
            if decision is not None and decision.load_skill and skill_round:
                text = self._skill_text(decision.load_skill)
                if text:
                    loaded.append(decision.load_skill)
                if text and not decision.actions:
                    msgs = msgs + [{"role": "assistant", "content": payload},
                                   {"role": "user", "content": text + "\n\nNow submit the decision."}]
                    skill_round = False
                    reply, mode = self._call(msgs, allow_skill=False)
                    continue
            break

        if decision is not None and (decision.actions or not problems):
            return decision, mode if mode != "text" else "json_text", payload, problems, loaded
        # One corrective retry with the validator's feedback.
        repair = msgs + [
            {"role": "assistant", "content": payload if isinstance(payload, str) else json.dumps(payload)},
            {"role": "user", "content": "That reply was not a valid decision: " + "; ".join(problems)
             + ". Reply again with only the decision object."},
        ]
        reply2, _ = self._call(repair, allow_skill=False)
        payload2 = (reply2.payload_for(TOOL_NAME) or reply2.content) if reply2 is not None else None
        decision2, problems2 = parse_decision(payload2, self.max_actions, self.n_action_ids) \
            if payload2 is not None else (None, ["LLM call failed"])
        if decision2 is not None and (decision2.actions or not problems2):
            return decision2, "repair", payload2, problems + problems2, loaded
        return None, "none", payload2 if payload2 is not None else payload, problems + problems2, loaded
