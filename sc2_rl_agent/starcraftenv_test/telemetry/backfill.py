"""
Backfill old log/<agent>_log/game_* folders into the Phase-0 schema.

    python -m sc2_rl_agent.starcraftenv_test.telemetry.backfill \\
        --log_root sc2_rl_agent/starcraftenv_test/log/chatgpt_log \\
        --exp_group legacy --wandb_mode online

Old games have no tokens, latency or slot sources. Slot sources are
re-derived by replaying the agent's queue: the same tick schedule
(current_step % action_interval), the same mix_actions placement, the
extractor re-run on commander.json for the parsed count, and
resource_rejections.json for filter drops. The run config is marked
backfilled=true, padding_inferred=true.

Telemetry for each game is written to <game_dir>/backfill/; the original
files are never modified.
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter, deque
from types import SimpleNamespace

from .funnel import build_tick, mix_with_sources, source_labels, match_failure, parse_failure, fail_cat
from .game_metrics import game_time_seconds
from .run_logger import RunLogger, FILES, SUMMARY_FILE
from .schema import SlotOutcome, EMPTY_ACTION_NAME, PARSE_TIERS

OLD_FILES = ("action_executed", "action_failures", "combined_input", "commander",
             "raw_observation", "resource_rejections")

_GAME_DIR_RE = re.compile(r"game_(\d{8}_\d{6})_(-?\d+)$")
_REPLAY_RE = re.compile(
    r"^\d{8}_\d{6}_(?P<map>.+?)_player_race_(?P<race>[A-Z]+)_VS_(?:BUILD_IN_AI_(?P<difficulty>[A-Za-z]+)"
    r"|RULE_AI_(?P<bot>.+?))_(?P<opp>[A-Za-z]+)_process_(?P<pid>-?\d+)_(?P<result>.+)\.SC2Replay$")


def _load_jsonl(path):
    import json
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append(None)
    return out


def _find_replay(game_dir):
    name = os.path.basename(os.path.normpath(game_dir))
    log_dir = os.path.dirname(os.path.dirname(os.path.abspath(game_dir)))
    candidates = glob.glob(os.path.join(game_dir, "*.SC2Replay"))
    candidates += glob.glob(os.path.join(log_dir, "*", name, "*.SC2Replay"))
    finals = [c for c in candidates if not c.endswith("_temp.SC2Replay")]
    return sorted(finals or candidates)[-1] if candidates else None


def replay_info(path):
    if not path:
        return {}
    m = _REPLAY_RE.match(os.path.basename(path))
    if not m:
        return {"replay_path": path}
    d = m.groupdict()
    info = {"replay_path": path, "map": d["map"], "opposite_race": d["opp"],
            "result": d["result"], "player_race": d["race"].capitalize()}
    if d["difficulty"]:
        info.update(opposite_type="build_in", difficulty=d["difficulty"])
    else:
        info.update(opposite_type="rule", opposite_bot=d["bot"])
    return info


class _StubVDB:
    """Stands in for the Chroma action DB: names that need a similarity
    lookup resolve to a placeholder id (-1) so they still count as parsed."""
    def search_actions(self, action):
        return {"ids": ["-1"], "documents": [action]}


def _command_text(rec):
    if isinstance(rec, list):
        return " ".join(str(x) for x in rec)
    return "" if rec is None else str(rec)


class _Parser:
    def __init__(self, race):
        from sc2_rl_agent.starcraftenv_test.utils.action_info import ActionDescriptions
        from sc2_rl_agent.starcraftenv_test.utils.action_extractor import ActionExtractor
        desc = ActionDescriptions(race)
        self.empty_idx = desc.empty_action_id
        self.id_to_name = dict(desc.flattened_actions)
        self.extractor = ActionExtractor(desc.action_descriptions)
        self.vdb = _StubVDB()

    def parse(self, command):
        from sc2_rl_agent.starcraftenv_test.utils.action_extractor import extract_actions_from_command
        info = {}
        ids, _ = extract_actions_from_command(command, self.extractor, self.empty_idx, self.vdb, info=info)
        return ids, info.get("parse_tier", "none")


def _apply_rejections(ids, id_to_name, rejected_names, empty_idx):
    """Rebuild the filter output: rejected names (as a multiset) become EMPTY."""
    remaining = Counter(rejected_names)
    kept = []
    for aid in ids:
        name = id_to_name.get(aid)
        if name is not None and remaining[name] > 0:
            remaining[name] -= 1
            kept.append(empty_idx)
        else:
            kept.append(aid)
    return kept


def _mismatches(slots, executed, tick_step):
    """How many slots disagree with what the env reported executing, given
    that slot p of this tick is dequeued at tick_step + p and its result is
    on line tick_step + p + 1."""
    bad = 0
    for p, (_, src) in enumerate(slots):
        i = tick_step + p + 1
        if i >= len(executed) or not isinstance(executed[i], str):
            continue
        bad += (src == "llm") != (executed[i] != EMPTY_ACTION_NAME)
    return bad


def backfill_game(game_dir, mode="disabled", exp_group="legacy", action_interval=10,
                  action_window=10, action_mix_rate=0.5, k=5, model_name=None, wandb_module=None):
    game_dir = os.path.abspath(game_dir)
    data = {stem: _load_jsonl(os.path.join(game_dir, f"{stem}.json")) for stem in OLD_FILES}
    executed, failures = data["action_executed"], data["action_failures"]
    raw_obs, commands = data["raw_observation"], data["commander"]
    rejections_by_step = {}
    for rec in data["resource_rejections"]:
        if isinstance(rec, dict) and "step" in rec:
            rejections_by_step[rec["step"]] = [r.get("action") for r in rec.get("rejections", [])]

    m = _GAME_DIR_RE.search(os.path.basename(game_dir))
    rinfo = replay_info(_find_replay(game_dir))
    race = rinfo.get("player_race", "Protoss")
    args = SimpleNamespace(
        agent_type="chatgpt",
        LLM_model_name=model_name or "unknown",
        player_race=race,
        opposite_race=rinfo.get("opposite_race"),
        opposite_type=rinfo.get("opposite_type"),
        difficulty=rinfo.get("difficulty", "unknown"),
        map=rinfo.get("map"),
        current_time=m.group(1) if m else os.path.basename(game_dir),
        process_id=m.group(2) if m else "-1",
        exp_group=exp_group,
        wandb_mode=mode,
    )
    out_dir = os.path.join(game_dir, "backfill")
    for fname in list(FILES.values()) + [SUMMARY_FILE]:
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            os.remove(p)
    logger = RunLogger(args, game_dir, mode=mode, out_dir=out_dir, job_type="backfill",
                       wandb_module=wandb_module, register_atexit=False, extra_config={
                           "backfilled": True, "padding_inferred": True,
                           "source_game_dir": game_dir, "prompt_type": "v2", "K": k,
                           "action_interval": action_interval, "action_window": action_window,
                           "action_mix_rate": action_mix_rate,
                           "resource_filter_enabled": bool(data["resource_rejections"]) or None,
                           "map": rinfo.get("map"),
                       })

    parser = _Parser(race)
    empty_idx = parser.empty_idx
    queue = deque()
    pending = None
    current_step = 0
    tick_idx = 0
    tick_id = 0
    game_time_s = 0.0
    inconsistent = 0

    for step in range(len(executed)):
        obs = raw_obs[step] if step < len(raw_obs) else None
        t = game_time_seconds(obs)
        if t is not None:
            game_time_s = t
        step_failures = failures[step] if step < len(failures) and isinstance(failures[step], list) else []

        # Result of the slot dequeued on the previous step.
        if pending is not None:
            src, tid, deq_step, deq_time = pending
            name = executed[step] if isinstance(executed[step], str) else None
            hit = None if name in (None, EMPTY_ACTION_NAME) else match_failure(name, step_failures)
            if src == "llm" and name == EMPTY_ACTION_NAME:
                inconsistent += 1
            elif src != "llm" and name not in (None, EMPTY_ACTION_NAME):
                inconsistent += 1
            logger.log_slot_outcome(SlotOutcome(
                slot_source=src, action_id=-1, action_name=name or EMPTY_ACTION_NAME,
                executed_name=name, failed=hit is not None,
                fail_reason=parse_failure(hit)[1] if hit else None,
                fail_cat=fail_cat(hit) if hit else None,
                tick_id=tid, dequeue_step=deq_step, game_time_s=game_time_s))
            pending = None

        logger.log_observation(step, game_time_s, obs, action_failures=step_failures)

        # ChatGPTAgent.action(): non-dict observations skip the tick and
        # do not advance current_step.
        if isinstance(obs, dict):
            if current_step % action_interval == 0 and tick_idx < len(commands):
                command = _command_text(commands[tick_idx])
                tick_idx += 1
                ids, tier = parser.parse(command)
                rejected = rejections_by_step.get(current_step, [])
                candidates = [list(ids)]
                if tier == "vectordb" and -1 in ids and not queue:
                    # The real vector DB may have returned nothing for a
                    # name; keep whichever reading matches what executed.
                    without = [a for a in ids if a != -1]
                    if without:
                        candidates.append(without)
                best = None
                for proposed in candidates:
                    kept = _apply_rejections(proposed, parser.id_to_name, rejected, empty_idx)
                    sources = source_labels(proposed, kept, tier, empty_idx)
                    slots = mix_with_sources(list(kept), sources, action_window, action_mix_rate, empty_idx)
                    score = _mismatches(slots, executed, step)
                    if best is None or score < best[0]:
                        best = (score, proposed, kept, slots)
                _, proposed, kept, slots = best
                logger.log_tick(build_tick(tick_id, None, step, game_time_s, k, tier, proposed, kept,
                                           slots, empty_idx, command_preview=command[:500]))
                queue.extend((src, tick_id) for _, src in slots)
                tick_id += 1
            current_step += 1

        if queue:
            src, tid = queue.popleft()
        else:
            src, tid = "queue_empty", None
        pending = (src, tid, step, game_time_s)

    if pending is not None:
        src, tid, deq_step, deq_time = pending
        logger.log_slot_outcome(SlotOutcome(
            slot_source=src, action_id=-1, action_name=EMPTY_ACTION_NAME, executed_name=None,
            failed=False, fail_reason=None, fail_cat=None, tick_id=tid, dequeue_step=deq_step,
            game_time_s=deq_time, resolved=False))

    return logger.finish(rinfo.get("result"), {
        "wall_clock_s": None,
        "wall_clock_share_waiting_llm": None,
        "backfilled": True,
        "backfill_inconsistent_slots": inconsistent,
    })


def main(argv=None):
    p = argparse.ArgumentParser(description="Backfill old game logs into Phase-0 telemetry.")
    p.add_argument("--log_root", required=True, help="Folder containing game_* directories.")
    p.add_argument("--games", nargs="*", default=None, help="Only these game_* folder names.")
    p.add_argument("--exp_group", default="legacy")
    p.add_argument("--wandb_mode", default=None, choices=["online", "offline", "disabled"])
    p.add_argument("--action_interval", type=int, default=10)
    p.add_argument("--action_window", type=int, default=10)
    p.add_argument("--action_mix_rate", type=float, default=0.5)
    p.add_argument("--k", type=int, default=5, help="K decisions requested per tick.")
    p.add_argument("--model_name", default=None, help="Model used by the old games, if known.")
    a = p.parse_args(argv)

    dirs = sorted(d for d in glob.glob(os.path.join(a.log_root, "game_*")) if os.path.isdir(d))
    if a.games:
        dirs = [d for d in dirs if os.path.basename(d) in set(a.games)]
    if not dirs:
        print(f"No game_* folders found under {a.log_root}")
        return 1
    for d in dirs:
        if not os.path.exists(os.path.join(d, "action_executed.json")):
            print(f"skip {os.path.basename(d)}: no action_executed.json")
            continue
        s = backfill_game(d, mode=a.wandb_mode, exp_group=a.exp_group,
                          action_interval=a.action_interval, action_window=a.action_window,
                          action_mix_rate=a.action_mix_rate, k=a.k, model_name=a.model_name)
        print(f"{os.path.basename(d)}: steps={s.get('n_steps')} ticks={s.get('n_ticks')} "
              f"result={s.get('result')} padding_share={s.get('padding_share'):.3f} "
              f"conversion={s.get('conversion'):.3f} exec_success={s.get('exec_success'):.3f} "
              f"inconsistent_slots={s.get('backfill_inconsistent_slots')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
