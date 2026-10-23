"""
Per-attempt LLM call metering.

The chatbot calls `meter.begin(messages)` once per logical query and then
`success()` / `error()` once per HTTP attempt. Nothing here may raise into
the caller: the chatbot wraps every meter call, and the meter itself only
reads response fields defensively.
"""

import time
from typing import Callable, List, Optional

from .schema import LLMCallRecord


def _get(obj, name, default=None):
    """Read a field from an openai>=1 pydantic object, its extras, or a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, None)
    if val is not None:
        return val
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and name in extra:
        return extra[name]
    return default


def _int_or_none(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def usage_fields(response):
    usage = _get(response, "usage")
    details = _get(usage, "completion_tokens_details")
    return {
        "prompt_tokens": _int_or_none(_get(usage, "prompt_tokens")) or 0,
        "completion_tokens": _int_or_none(_get(usage, "completion_tokens")) or 0,
        "reasoning_tokens": _int_or_none(_get(details, "reasoning_tokens")),
        # DeepSeek's context-cache fields; absent on other providers.
        "cache_hit_tokens": _int_or_none(_get(usage, "prompt_cache_hit_tokens")),
        "cache_miss_tokens": _int_or_none(_get(usage, "prompt_cache_miss_tokens")),
    }


class MeteredCall:
    def __init__(self, meter, call_id, messages):
        self.meter = meter
        self.call_id = call_id
        self.messages = messages
        self.t_begin = meter.clock()
        self.failed = False

    def _base(self, attempt, t0):
        ctx = self.meter.context
        return dict(
            call_id=self.call_id,
            game_step=ctx["game_step"],
            game_time_s=ctx["game_time_s"],
            purpose=ctx["purpose"],
            model_requested=self.meter.model_requested,
            reasoning_effort=self.meter.reasoning_effort,
            latency_s=round(self.meter.clock() - t0, 4),
            retries=attempt,
        )

    def success(self, attempt, t0, response, answer):
        choice0 = None
        choices = _get(response, "choices")
        if choices:
            choice0 = choices[0]
        message = _get(choice0, "message")
        reasoning = _get(message, "reasoning_content")
        rec = LLMCallRecord(
            **self._base(attempt, t0),
            model_returned=_get(response, "model"),
            **usage_fields(response),
            error=None,
            finish_reason=_get(choice0, "finish_reason"),
            output_chars=len(answer) if isinstance(answer, str) else 0,
            has_reasoning_content=bool(reasoning),
        )
        self._finish(rec, answer, reasoning)

    def error(self, attempt, t0, exc, final):
        rec = LLMCallRecord(
            **self._base(attempt, t0),
            model_returned=None,
            prompt_tokens=0, completion_tokens=0, reasoning_tokens=None,
            cache_hit_tokens=None, cache_miss_tokens=None,
            error=f"{type(exc).__name__}: {exc}"[:2000],
            finish_reason=None,
            output_chars=0,
            has_reasoning_content=False,
            final_failure=bool(final),
        )
        self.failed = bool(final)
        self._finish(rec, None, None, last=final)

    def _finish(self, rec, answer, reasoning, last=True):
        if self.meter.sink is not None:
            self.meter.sink(rec, messages=self.messages, response_text=answer,
                            reasoning_content=reasoning)
        if last:
            info = {
                "call_id": self.call_id,
                "failed": self.failed,
                # Includes the backoff sleeps between retries.
                "wall_s": self.meter.clock() - self.t_begin,
                "attempts": rec.retries + 1,
            }
            self.meter.completed.append(info)
            if self.meter.on_complete is not None:
                self.meter.on_complete(info)


class LLMMeter:
    def __init__(self, sink: Optional[Callable] = None, model_requested="", reasoning_effort=None,
                 clock=time.perf_counter, on_complete: Optional[Callable] = None):
        self.sink = sink
        self.on_complete = on_complete
        self.model_requested = model_requested
        self.reasoning_effort = reasoning_effort
        self.clock = clock
        self.context = {"game_step": -1, "game_time_s": 0.0, "purpose": "unknown"}
        self.completed: List[dict] = []
        self._next_id = 0

    def set_context(self, game_step, game_time_s, purpose):
        self.context = {"game_step": game_step, "game_time_s": game_time_s, "purpose": purpose}

    def begin(self, messages) -> MeteredCall:
        call = MeteredCall(self, self._next_id, messages)
        self._next_id += 1
        return call

    def pop_completed(self) -> List[dict]:
        out, self.completed = self.completed, []
        return out
