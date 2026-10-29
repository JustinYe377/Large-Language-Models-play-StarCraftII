"""
RunLogger: local JSONL is the source of truth, wandb is an optional sink.

Every log_* call writes its JSONL line first and only then touches wandb.
Any wandb exception is reported once and wandb is switched off for the rest
of the run, so a missing login, a network drop, or a wandb bug can never
stop a game or lose local data.

wandb is imported lazily inside __init__ (never at module import) so that
forked SC2 bot processes never load it.
"""

import atexit
import glob
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from collections import Counter, deque

import numpy as np

from .schema import SCHEMA_VERSION, EMPTY_ACTION_NAME, ATTACK_ACTION_NAME, SCRIPTED_ATTACK_KEYS
from .funnel import FunnelCounter, fail_cat
from .game_metrics import extract_metrics, GameMetricsTracker

WANDB_MODES = ("online", "offline", "disabled")
DEFAULT_MODE = "offline"
PROJECT = "textsc2"
GAME_LOG_EVERY_S = 5.0
REDACTED = "***REDACTED***"
COMMAND_PREVIEW_CHARS = 500

# Placeholder defaults that are not real secrets.
_FAKE_KEYS = {"", "your-api-key", "your-key", "none", "empty"}
# Keys shorter than this (e.g. "x" or "token" for a local vLLM/Ollama server)
# are not treated as secrets: scrubbing them would corrupt every log line.
MIN_SECRET_LEN = 8

FILES = {
    "llm_call": "llm_calls.jsonl",
    "tick": "decision_ticks.jsonl",
    "slot": "slots.jsonl",
    "observation": "observations.jsonl",
    "event": "events.jsonl",
}
SUMMARY_FILE = "game_summary.json"


def resolve_mode(cli_mode=None):
    """CLI flag > WANDB_MODE env > offline."""
    mode = cli_mode or os.environ.get("WANDB_MODE") or DEFAULT_MODE
    mode = str(mode).lower()
    if mode == "dryrun":
        mode = "offline"
    if mode not in WANDB_MODES:
        warnings.warn(f"[telemetry] unknown wandb mode {mode!r}; using disabled")
        mode = "disabled"
    return mode


NORMAL_RESULTS = ("Victory", "Defeat", "Tie")


def exit_code_for(result):
    """0 for a finished game, 130 (SIGINT convention) for Ctrl+C, else 1."""
    if result in NORMAL_RESULTS:
        return 0
    return 130 if result == "aborted" else 1


def normalize_result(result):
    if result is None:
        return "unknown"
    text = str(result)
    return text.split(".", 1)[1] if text.startswith("Result.") else text


def _git_info(cwd):
    def run(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=cwd, stderr=subprocess.DEVNULL,
                                           timeout=10).decode().strip()
        except Exception:
            return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _short(text):
    return re.sub(r"[^A-Za-z0-9]", "", str(text).replace(" LE", ""))


def _percentile(values, q):
    return float(np.percentile(values, q)) if values else None


class RunLogger:
    def __init__(self, args, game_folder, mode=None, *, out_dir=None, job_type="game",
                 extra_config=None, wandb_module=None, register_atexit=True):
        self.args = args
        self.game_folder = os.path.abspath(game_folder)
        self.out_dir = os.path.abspath(out_dir or game_folder)
        os.makedirs(self.out_dir, exist_ok=True)
        self.mode = resolve_mode(mode)
        self.job_type = job_type
        self.t_start = time.time()
        self.finished = False

        key = getattr(args, "LLM_api_key", None)
        self._secrets = ([key] if isinstance(key, str) and len(key.strip()) >= MIN_SECRET_LEN
                         and key.strip().lower() not in _FAKE_KEYS else [])

        self._files = {}
        self.funnel = FunnelCounter(rolling_ticks=10)
        self.game = GameMetricsTracker()
        self.n_steps = 0
        self.fail_cats = Counter()
        self.action_counts = Counter()
        self.failure_rows = []
        self.tick_rows = []
        self.llm_records = []
        self.llm_call_infos = []
        self._recent_latency = deque(maxlen=10)
        self.model_resolved = None
        self.time_to_first_attack_s = None
        self.scripted_attack = {k: 0 for k in SCRIPTED_ATTACK_KEYS}
        self.scripted_attack_first_order_s = None
        self.replay_path = None
        self._last_game_log_t = None
        self._last_game_time_s = 0.0
        self.summary_extra = {}          # agent-provided fields merged into the summary

        self.config = self._build_config(extra_config or {})
        self.run_name = self._run_name()
        self._write("event", "run_start", {"run_name": self.run_name, "mode": self.mode,
                                            "config": self.config})

        self._wandb = None
        self._run = None
        self._wandb_warned = False
        if self.mode != "disabled":
            self._init_wandb(wandb_module)
        if register_atexit:
            atexit.register(self._atexit)

    # ------------------------------------------------------------------
    # config / naming
    # ------------------------------------------------------------------

    def _build_config(self, extra):
        cfg = {}
        for k, v in vars(self.args).items():
            if k in ("LLM_api_key", "run_logger") or "api_key" in k.lower():
                continue
            cfg[k] = _jsonable(v)
        cfg.update(_git_info(os.path.dirname(os.path.abspath(__file__))))
        cfg.update({
            "model_requested": getattr(self.args, "LLM_model_name", None),
            "model_resolved": None,
            "reasoning_effort": getattr(self.args, "LLM_reasoning_effort", None),
            "temperature": getattr(self.args, "LLM_temperature", None),
            "burnysc2_version": _pkg_version("burnysc2"),
            "openai_version": _pkg_version("openai"),
            "sc2_version": None,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "schema_version": SCHEMA_VERSION,
        })
        try:
            from ..env.bot.version import BOT_VERSION
            cfg["bot_version"] = BOT_VERSION
        except Exception:
            cfg["bot_version"] = None
        if self.job_type == "game":
            # Which SC2 build/platform this machine launches (headless Linux 4.10
            # vs Windows 5.0.x give different balance, so results must not mix).
            try:
                from ..sc2_runtime import runtime_info
                cfg.update(runtime_info())
            except Exception as e:
                print(f"[telemetry] could not read SC2 runtime info: {e}")
        map_pool, map_idx = getattr(self.args, "map_pool", None), getattr(self.args, "map_idx", None)
        if "map" not in extra and isinstance(map_pool, (list, tuple)) and isinstance(map_idx, int):
            try:
                cfg["map"] = map_pool[map_idx]
            except IndexError:
                pass
        cfg.update(_jsonable(extra))
        return self._redact_obj(cfg)

    def _run_name(self):
        model = str(self.config.get("model_requested") or "model").split("/")[-1]
        return "-".join([
            str(self.config.get("difficulty", "na")),
            _short(self.config.get("map", "map")),
            model,
            str(self.config.get("current_time", time.strftime("%Y%m%d_%H%M%S"))),
            f"p{self.config.get('process_id', '-1')}",
        ])

    def _tags(self):
        tags = [self.config.get("agent_type"), self.config.get("difficulty"),
                self.config.get("opposite_race"), self.config.get("prompt_type"), "phase0"]
        if self.job_type == "backfill":
            tags.append("backfill")
        return [str(t) for t in tags if t]

    # ------------------------------------------------------------------
    # local JSONL
    # ------------------------------------------------------------------

    def _redact(self, text):
        for s in self._secrets:
            text = text.replace(s, REDACTED)
        return text

    def _redact_obj(self, obj):
        if not self._secrets:
            return obj
        return json.loads(self._redact(json.dumps(obj, ensure_ascii=False)))

    def _write(self, kind, type_, payload):
        try:
            f = self._files.get(kind)
            if f is None:
                f = open(os.path.join(self.out_dir, FILES[kind]), "a", buffering=1, encoding="utf-8")
                self._files[kind] = f
            line = {"schema_version": SCHEMA_VERSION, "type": type_, "ts": time.time()}
            line.update(payload)
            f.write(self._redact(json.dumps(_jsonable(line), ensure_ascii=False)) + "\n")
        except Exception as e:
            print(f"[telemetry] failed to write {kind} record: {e}")

    # ------------------------------------------------------------------
    # wandb sink
    # ------------------------------------------------------------------

    def _init_wandb(self, wandb_module):
        try:
            if wandb_module is None:
                import wandb as wandb_module
            self._wandb = wandb_module
            self._run = wandb_module.init(
                project=PROJECT,
                group=getattr(self.args, "exp_group", None) or "adhoc",
                job_type=self.job_type,
                name=self.run_name,
                tags=self._tags(),
                config=self.config,
                mode=self.mode,
            )
            self._run.define_metric("game_time_s")
            for prefix in ("game/*", "funnel/*", "llm/*"):
                self._run.define_metric(prefix, step_metric="game_time_s")
        except Exception as e:
            self._wandb_failed(e)

    def _wandb_failed(self, e):
        self._warn_wandb(e)
        self._run = None
        self._wandb = None

    def _warn_wandb(self, e):
        if not self._wandb_warned:
            self._wandb_warned = True
            msg = f"[telemetry] wandb error, continuing with local JSONL only: {type(e).__name__}: {e}"
            print(msg)
            warnings.warn(msg)
            self._write("event", "wandb_disabled", {"error": f"{type(e).__name__}: {e}"})

    def _wb(self, fn):
        if self._run is None:
            return
        try:
            fn(self._run)
        except Exception as e:
            self._wandb_failed(e)

    @property
    def wandb_active(self):
        return self._run is not None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def log_llm_call(self, rec, messages=None, response_text=None, reasoning_content=None):
        payload = rec.to_dict()
        payload.update({"messages": messages, "response": response_text,
                        "reasoning_content": reasoning_content})
        self._write("llm_call", "llm_call", payload)
        self.llm_records.append(rec)
        if rec.error is None and rec.model_returned:
            if self.model_resolved is None:
                self.model_resolved = rec.model_returned
                self.config["model_resolved"] = rec.model_returned
                self._wb(lambda r: r.config.update({"model_resolved": rec.model_returned},
                                                   allow_val_change=True))
            elif rec.model_returned != self.model_resolved:
                self.log_event("model_changed", level="warning", previous=self.model_resolved,
                               current=rec.model_returned, call_id=rec.call_id)
                self.model_resolved = rec.model_returned

    def log_llm_call_complete(self, info):
        self.llm_call_infos.append(info)

    def log_tick(self, tick):
        self._write("tick", "decision_tick", tick.to_dict())
        self.funnel.add_tick(tick)
        self.tick_rows.append(tick)

        attempts = [r for r in self.llm_records if r.call_id == tick.call_id] if tick.call_id is not None else []
        data = {"game_time_s": tick.game_time_s}
        data.update({f"funnel/{k}": v for k, v in self.funnel.rolling().items()})
        data.update({
            "funnel/tick_parse_rate": tick.n_parsed / tick.k_requested if tick.k_requested else 0.0,
            "funnel/tick_n_queued_real": tick.n_queued_real,
            "funnel/tick_n_padding": tick.n_padding,
            "funnel/tick_n_filter_drop_slots": tick.n_filter_drop_slots,
            "funnel/tick_n_nonreal_slots": tick.n_nonreal_slots,
            "funnel/tick_n_truncated": tick.n_truncated,
        })
        if attempts:
            ok = [r for r in attempts if r.error is None]
            latency = sum(r.latency_s for r in attempts)
            self._recent_latency.append(latency)
            hit = sum(r.cache_hit_tokens or 0 for r in ok)
            miss = sum(r.cache_miss_tokens or 0 for r in ok)
            data.update({
                "llm/latency_s": latency,
                "llm/latency_s_rolling_mean": float(np.mean(self._recent_latency)),
                "llm/prompt_tokens": sum(r.prompt_tokens for r in ok),
                "llm/completion_tokens": sum(r.completion_tokens for r in ok),
                "llm/reasoning_tokens": sum(r.reasoning_tokens or 0 for r in ok),
                "llm/cache_hit_ratio": hit / (hit + miss) if (hit + miss) else 0.0,
                "llm/retries": len(attempts) - 1,
                "llm/error": int(tick.llm_error),
            })
        self._wb(lambda r: r.log(data))

    def log_slot_outcome(self, o):
        payload = o.to_dict()
        payload["resolve_step"] = o.dequeue_step + 1 if o.resolved else None
        self._write("slot", "slot_outcome", payload)
        self.funnel.add_outcome(o)
        if not o.resolved:
            return
        if o.failed:
            self.failure_rows.append([o.dequeue_step + 1, o.game_time_s, o.slot_source,
                                      o.action_name, o.fail_reason, o.fail_cat])
        elif o.action_name and o.action_name != EMPTY_ACTION_NAME:
            self.action_counts[o.action_name] += 1
            if o.action_name == ATTACK_ACTION_NAME and self.time_to_first_attack_s is None:
                self.time_to_first_attack_s = o.game_time_s

    def log_observation(self, step, game_time_s, raw_obs, action_failures=None, extra=None):
        self.n_steps += 1
        metrics = extract_metrics(raw_obs) if isinstance(raw_obs, dict) else {}
        failures = action_failures if isinstance(action_failures, list) else []
        for f in failures:
            self.fail_cats[fail_cat(str(f))] += 1
        extra = dict(extra or {})
        self.update_scripted_attack(extra, game_time_s)
        version = extra.pop("sc2_game_version", None)
        if version and self.config.get("sc2_version") != version:
            self.config["sc2_version"] = version
            self._wb(lambda r: r.config.update({"sc2_version": version}, allow_val_change=True))
        if extra.get("replay_path"):
            self.replay_path = extra["replay_path"]

        self._write("observation", "observation", {
            "step": step, "game_time_s": game_time_s, "metrics": metrics,
            "n_failures": len(failures), **extra,
        })
        if game_time_s is None:
            return
        self._last_game_time_s = game_time_s
        self.game.update(game_time_s, metrics)
        if metrics and (self._last_game_log_t is None or
                        game_time_s - self._last_game_log_t >= GAME_LOG_EVERY_S):
            self._last_game_log_t = game_time_s
            data = {"game_time_s": game_time_s}
            data.update({f"game/{k}": v for k, v in metrics.items()})
            self._wb(lambda r: r.log(data))

    def update_scripted_attack(self, values, game_time_s=None):
        """Counters are cumulative in the bot, so keep the max seen."""
        for k in SCRIPTED_ATTACK_KEYS:
            v = values.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                self.scripted_attack[k] = max(self.scripted_attack[k], v)
        if self.scripted_attack["scripted_attack_order_steps"] > 0 and self.scripted_attack_first_order_s is None:
            self.scripted_attack_first_order_s = game_time_s

    def log_event(self, kind, **payload):
        self._write("event", kind, payload)
        if payload.get("level") == "warning":
            print(f"[telemetry] warning: {kind} {payload}")

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def _price(self, name):
        try:
            return float(getattr(self.args, name, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def build_summary(self, result, extra):
        ok = [r for r in self.llm_records if r.error is None]
        latencies = [r.latency_s for r in self.llm_records]
        hit_vals = [r.cache_hit_tokens for r in ok if r.cache_hit_tokens is not None]
        miss_vals = [r.cache_miss_tokens for r in ok if r.cache_miss_tokens is not None]
        tokens_prompt = sum(r.prompt_tokens for r in ok)
        tokens_completion = sum(r.completion_tokens for r in ok)
        hit = sum(hit_vals) if hit_vals else None
        miss = sum(miss_vals) if miss_vals else None
        wall = time.time() - self.t_start

        p_hit, p_miss, p_out = (self._price("price_in_hit"), self._price("price_in_miss"),
                                self._price("price_out"))
        est_cost = None
        if p_hit or p_miss or p_out:
            if hit is not None or miss is not None:
                cost_in = (hit or 0) * p_hit + (miss or 0) * p_miss
            else:
                cost_in = tokens_prompt * p_miss
            est_cost = (cost_in + tokens_completion * p_out) / 1e6

        call_ids = {r.call_id for r in self.llm_records}
        llm_wait = sum(i["wall_s"] for i in self.llm_call_infos) or sum(latencies)

        s = {
            "result": normalize_result(result),
            "game_duration_s": self.game.last_game_time_s or self._last_game_time_s,
            "wall_clock_s": round(wall, 3),
            "n_steps": self.n_steps,
            "n_ticks": len(self.tick_rows),
            "n_llm_calls": len(call_ids),
            "n_llm_attempts": len(self.llm_records),
            "n_llm_errors": sum(1 for r in self.llm_records if r.error is not None),
            "n_llm_failed_calls": sum(1 for r in self.llm_records if r.final_failure),
            "total_retries": len(self.llm_records) - len(call_ids),
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_reasoning": sum(r.reasoning_tokens or 0 for r in ok),
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "cache_hit_ratio": (hit / (hit + miss)) if hit is not None and miss is not None and (hit + miss) else None,
            "est_cost_usd": est_cost,
            "latency_p50": _percentile(latencies, 50),
            "latency_p95": _percentile(latencies, 95),
            "wall_clock_share_waiting_llm": (llm_wait / wall) if wall > 0 else None,
            "model_requested": self.config.get("model_requested"),
            "model_resolved": self.model_resolved,
            "sc2_version": self.config.get("sc2_version"),
            "sc2_platform": self.config.get("sc2_platform"),
            "bot_version": self.config.get("bot_version"),
            "bot_rules": self.config.get("bot_rules"),
            "time_to_first_attack_s": self.time_to_first_attack_s,
            **self.scripted_attack,
            "scripted_attack_first_order_s": self.scripted_attack_first_order_s,
            "replay_path": self.replay_path,
            "schema_version": SCHEMA_VERSION,
        }
        s.update(self.funnel.totals())
        s.update(self.game.summary())
        for cat, n in self.fail_cats.items():
            s[f"fail_cat_{cat.replace(' ', '_')}"] = n
        for name, n in self.action_counts.items():
            s[f"action_count_{name.replace(' ', '_')}"] = n
        s.update(_jsonable(self.summary_extra))
        s.update(_jsonable(extra or {}))
        return s

    def _artifact_files(self):
        files = set()
        for pattern in ("*.jsonl", "*.json", "*.SC2Replay"):
            files.update(glob.glob(os.path.join(self.game_folder, pattern)))
            files.update(glob.glob(os.path.join(self.out_dir, pattern)))
        if self.replay_path and os.path.isfile(self.replay_path):
            files.add(os.path.abspath(self.replay_path))
        return sorted(files)

    def _stage_redacted(self, paths):
        """Files that contain the API key are uploaded as scrubbed copies kept
        under <out_dir>/artifact_redacted/; all other files are uploaded in place."""
        staged = []
        secrets = [s.encode() for s in self._secrets]
        red_dir = os.path.join(self.out_dir, "artifact_redacted")
        for p in paths:
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if secrets and any(s in data for s in secrets):
                os.makedirs(red_dir, exist_ok=True)
                for s in secrets:
                    data = data.replace(s, REDACTED.encode())
                dst = os.path.join(red_dir, os.path.basename(p))
                with open(dst, "wb") as f:
                    f.write(data)
                staged.append((dst, os.path.basename(p)))
                self.log_event("artifact_redacted", file=os.path.basename(p))
            else:
                staged.append((p, os.path.relpath(p, self.game_folder)))
        return staged

    def finish(self, result=None, extra=None):
        if self.finished:
            return None
        self.finished = True
        extra = dict(extra or {})
        if extra.get("replay_path"):
            self.replay_path = extra["replay_path"]
        if not self.replay_path:
            replays = sorted(glob.glob(os.path.join(self.game_folder, "*.SC2Replay")))
            self.replay_path = replays[-1] if replays else None

        summary = {}
        try:
            summary = self.build_summary(result, extra)
            with open(os.path.join(self.out_dir, SUMMARY_FILE), "w", encoding="utf-8") as f:
                f.write(self._redact(json.dumps(summary, indent=2, ensure_ascii=False)))
            self._write("event", "run_end", {"result": summary.get("result")})
        except Exception as e:
            print(f"[telemetry] failed to build game summary: {e}")
        for f in self._files.values():
            try:
                f.flush()
            except Exception:
                pass

        # Each step is attempted on its own so a failing table upload still
        # lets the artifact and run.finish() go through.
        run = self._run
        if run is not None:
            result = summary.get("result")
            for step in (lambda r: r.summary.update(summary), lambda r: self._tag_result(r, result),
                         self._log_tables, self._log_artifact,
                         lambda r: r.finish(exit_code=exit_code_for(result))):
                try:
                    step(run)
                except Exception as e:
                    self._warn_wandb(e)
        self._run = None
        self._wandb = None
        self.close()
        return summary

    @staticmethod
    def _tag_result(run, result):
        """wandb has no "aborted" state, so abnormal endings are also added as
        a run tag (aborted / crash / unknown) to make them filterable."""
        if result not in NORMAL_RESULTS:
            run.tags = tuple(run.tags or ()) + (str(result),)

    def _log_tables(self, run):
        wb = self._wandb
        tick_cols = ["tick_id", "game_step", "game_time_s", "call_id", "parse_tier", "k_requested",
                     "n_parsed", "n_unique_parsed", "n_after_filter", "n_filter_dropped",
                     "n_queued_real", "n_padding", "n_filter_drop_slots", "n_parse_empty_slots",
                     "n_nonreal_slots", "n_truncated", "llm_error",
                     "proposed_action_ids", "queued_action_ids", "command_preview"]
        ticks = wb.Table(columns=tick_cols, data=[
            [str(getattr(t, c)) if isinstance(getattr(t, c), list) else getattr(t, c) for c in tick_cols]
            for t in self.tick_rows])
        failures = wb.Table(columns=["resolve_step", "game_time_s", "slot_source", "action",
                                     "reason", "fail_cat"], data=self.failure_rows)
        hist = wb.Table(columns=["action", "count"],
                        data=[[k, v] for k, v in self.action_counts.most_common()])
        run.log({"decision_ticks": ticks, "failures": failures, "action_histogram": hist})

    def _log_artifact(self, run):
        name = re.sub(r"[^A-Za-z0-9._-]", "_", f"game-{self.run_name}")
        art = self._wandb.Artifact(name, type="game_logs")
        for path, arcname in self._stage_redacted(self._artifact_files()):
            art.add_file(path, name=arcname)
        run.log_artifact(art)

    def close(self):
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files = {}

    def _atexit(self):
        if not self.finished:
            try:
                self.finish(result="unknown", extra={"finish_reason": "atexit"})
            except Exception:
                pass
