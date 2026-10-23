"""
Glue between ChatGPTAgent and RunLogger.

The agent only calls the hook methods below. Each hook swallows its own
exceptions (reported once), so telemetry can never change what the agent
does or stop a game.
"""

import functools
import os

from .funnel import build_tick, resolve_slot
from .game_metrics import game_time_seconds
from .llm_meter import LLMMeter
from .run_logger import RunLogger, resolve_mode, COMMAND_PREVIEW_CHARS


def _safe(default=None):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not self.enabled:
                return default
            try:
                return fn(self, *args, **kwargs)
            except Exception as e:
                if not self._warned:
                    self._warned = True
                    print(f"[telemetry] {fn.__name__} failed, telemetry error ignored: "
                          f"{type(e).__name__}: {e}")
                return default
        return wrapper
    return deco


def _command_text(command):
    if isinstance(command, list):
        return " ".join(str(c) for c in command)
    return "" if command is None else str(command)


class AgentTelemetry:
    def __init__(self, logger=None, meter=None, k_requested=0, empty_idx=None, id_to_name=None):
        self.enabled = logger is not None
        self._warned = False
        self.logger = logger
        self.meter = meter
        self.k_requested = k_requested
        self.empty_idx = empty_idx
        self.id_to_name = id_to_name or {}
        self.step = -1
        self.game_time_s = 0.0
        self.next_tick_id = 0
        self.pending = None

    @classmethod
    def create(cls, agent, args, extra_config):
        """Build telemetry for a ChatGPTAgent. Never raises; returns a
        disabled instance if anything goes wrong."""
        try:
            logger = getattr(args, "run_logger", None)
            if logger is None:
                if hasattr(args, "wandb_mode"):
                    mode = resolve_mode(args.wandb_mode)
                else:
                    # Entry points that predate telemetry (multiprocess_test,
                    # worker.py) only write JSONL unless WANDB_MODE is set.
                    mode = resolve_mode(os.environ.get("WANDB_MODE") or "disabled")
                logger = RunLogger(args, agent.game_folder, mode=mode, extra_config=extra_config)
            meter = LLMMeter(sink=logger.log_llm_call, on_complete=logger.log_llm_call_complete,
                             model_requested=agent.model_name,
                             reasoning_effort=getattr(args, "LLM_reasoning_effort", None))
            agent.L2.chatbot.meter = meter
            k = getattr(args, "prompt_K", None)
            return cls(logger, meter, int(k) if k is not None else 5,
                       agent.empty_action_idx, agent._id_to_name)
        except Exception as e:
            print(f"[telemetry] disabled, could not initialise: {type(e).__name__}: {e}")
            return cls()

    # ------------------------------------------------------------------

    def _name(self, action_id):
        return self.id_to_name.get(action_id, str(action_id))

    def _resolve_pending(self, observation):
        if self.pending is None:
            return
        action_id, source, tick_id, step, _ = self.pending
        self.pending = None
        self.logger.log_slot_outcome(resolve_slot(
            action_id, self._name(action_id), source, tick_id, step, self.game_time_s,
            observation.get("action_executed"), observation.get("action_failures")))

    @_safe()
    def on_observation(self, observation):
        """Call first thing in action(): resolves the previous slot and
        records this step's game metrics."""
        self.step += 1
        raw = observation.get("information")
        t = game_time_seconds(raw, observation.get("game_loop"))
        if t is not None:
            self.game_time_s = t
        self._resolve_pending(observation)
        extra = {k: observation.get(k) for k in ("game_loop", "scripted_attack_count", "replay_path")
                 if observation.get(k) is not None}
        self.logger.log_observation(self.step, self.game_time_s, raw,
                                    action_failures=observation.get("action_failures"), extra=extra)

    @_safe()
    def before_llm(self, purpose="L2_actor"):
        self.meter.pop_completed()
        self.meter.set_context(self.step, self.game_time_s, purpose)

    @_safe()
    def on_tick(self, command, parse_tier, proposed_ids, kept_ids, slots):
        """Returns the tick_id to attach to this tick's queue slots."""
        completed = self.meter.pop_completed()
        tick_id = self.next_tick_id
        self.next_tick_id += 1
        tick = build_tick(
            tick_id=tick_id,
            call_id=completed[-1]["call_id"] if completed else None,
            game_step=self.step,
            game_time_s=self.game_time_s,
            k_requested=self.k_requested,
            parse_tier=parse_tier or "none",
            proposed_ids=proposed_ids,
            kept_ids=kept_ids,
            slots=slots,
            empty_idx=self.empty_idx,
            llm_error=any(c["failed"] for c in completed),
            command_preview=_command_text(command)[:COMMAND_PREVIEW_CHARS],
        )
        self.logger.log_tick(tick)
        return tick_id

    @_safe()
    def on_dequeue(self, action_id, slot_source, tick_id):
        self.pending = (action_id, slot_source, tick_id, self.step, self.game_time_s)

    @_safe()
    def finish(self, result, final_observation=None):
        extra = {}
        if isinstance(final_observation, dict):
            self._resolve_pending(final_observation)
            for k in ("replay_path", "scripted_attack_count", "game_loop"):
                if final_observation.get(k) is not None:
                    extra[k] = final_observation[k]
            if isinstance(extra.get("scripted_attack_count"), int):
                self.logger.scripted_attack_count = max(self.logger.scripted_attack_count,
                                                        extra["scripted_attack_count"])
        elif self.pending is not None:
            action_id, source, tick_id, step, t = self.pending
            self.pending = None
            o = resolve_slot(action_id, self._name(action_id), source, tick_id, step, t, None, [])
            o.resolved = False
            self.logger.log_slot_outcome(o)
        return self.logger.finish(result, extra)
