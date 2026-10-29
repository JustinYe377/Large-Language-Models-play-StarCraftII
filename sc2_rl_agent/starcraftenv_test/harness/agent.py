"""
HarnessAgent: structured-output agent with a compact, cache-friendly prompt
and action masking.

Phase 1 keeps the old decision cadence on purpose (a decision every
`action_interval` steps, mix_actions padding, the resource filter), so a
run compares directly against ChatGPTAgent in wandb. What changes:

  input   one state snapshot + changes + outcomes of the last actions + the
          standing plan, instead of five repeated L1 summaries
  output  a schema-validated decision (tool call / JSON) instead of an essay
          parsed with regexes; no hidden reasoning by default
  mask    only actions whose prerequisites exist are legal; others are
          discarded and reported back to the model

It writes the same legacy log files as ChatGPTAgent so sc2_llm_analyzer.py
keeps working, plus the usual telemetry JSONL.
"""

import json
import os
from collections import Counter, deque

from sc2_rl_agent.starcraftenv_test.summarize.L1_summarize import generate_summarize_L1
from sc2_rl_agent.starcraftenv_test.telemetry.agent_hooks import AgentTelemetry
from sc2_rl_agent.starcraftenv_test.telemetry.funnel import mix_with_sources, source_labels, parse_failure
from sc2_rl_agent.starcraftenv_test.utils.path_utils import agent_log_dir_name
from sc2_rl_agent.starcraftenv_test.utils.resource_filter import (
    parse_resources, filter_by_resources, format_rejections_for_actor,
)

from .actions import PROTOSS_ACTIONS, legal_actions
from .llm import StructuredClient
from .prompt import build_system_prompt, build_user_message, active_skill_message, goal_progress_lines
from .skills import SkillLibrary
from .state import parse_state, render_state, diff_states

HARNESS_VERSION = "v2-skills"
EMPTY_NAME = "EMPTY ACTION"


class HarnessAgent:
    def __init__(self, model_name, api_key, api_base, temperature, args, action_description,
                 action_interval=10, action_window=10, action_mix_rate=0.5, max_actions=None):
        if args.player_race != "Protoss":
            raise ValueError("HarnessAgent currently supports Protoss only")
        self.args = args
        self.model_name = model_name
        self.action_interval = action_interval
        self.action_window = action_window
        self.action_mix_rate = action_mix_rate
        self.max_actions = max_actions or getattr(args, "harness_max_actions", 5)
        self.action_description = action_description
        self.action_dict = action_description.action_descriptions
        self._id_to_name = dict(action_description.flattened_actions)
        self.empty_action_idx = action_description.empty_action_id
        self.current_step = 0
        self.action_queue = deque()
        self.process_id = args.process_id

        log_root = getattr(args, 'log_root', None) or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'log')
        base = os.path.join(log_root, agent_log_dir_name(getattr(args, 'agent_type', 'harness')))
        self.game_folder = os.path.join(base, f"game_{args.current_time}_{self.process_id}")
        os.makedirs(self.game_folder, exist_ok=True)

        self.skills = (SkillLibrary.for_opponent(args.opposite_race)
                       if getattr(args, "harness_skills", "on") != "off" else SkillLibrary({}))
        self.llm_client = StructuredClient(
            model_name, api_key, api_base, temperature, self.max_actions, len(self._id_to_name),
            mode=getattr(args, "structured_mode", "auto"),
            thinking=getattr(args, "LLM_thinking", "disabled"),
            thinking_provider=getattr(args, "LLM_thinking_provider", "auto"),
            skills=self.skills or None)
        self.llm_client.reasoning_effort = getattr(args, "LLM_reasoning_effort", None)
        opponent = (f"{args.difficulty} {args.opposite_race}" if args.opposite_type == 'build_in'
                    else f"{args.opposite_bot} {args.opposite_race}")
        self.system_prompt = build_system_prompt(self.max_actions, opponent,
                                                 skill_index=self.skills.index_text() if self.skills else "",
                                                 json_load_field=bool(self.skills))
        self.active_skill = None
        self.skills_loaded = []
        self.army_goal = None
        self.enemy_seen = {}

        self.bot_rules = getattr(args, "bot_rules", "dynamic")
        self.plan = None
        self.prev_state = None
        self._executed = Counter()        # since the last decision
        self._failed = Counter()          # (name, reason) -> n
        self._carry_lines = []            # filter / mask notes from the last decision

        self.telemetry = AgentTelemetry.create(self, args, extra_config={
            "harness_version": HARNESS_VERSION,
            "prompt_class": "harness.prompt",
            "prompt_type": "harness",
            "K": self.max_actions,
            "action_interval": action_interval,
            "action_window": action_window,
            "action_mix_rate": action_mix_rate,
            "resource_filter_enabled": True,
            "action_masking": True,
            "structured_mode": getattr(args, "structured_mode", "auto"),
            "thinking": getattr(args, "LLM_thinking", "disabled"),
            "skills": self.skills.names,
            "game_folder": self.game_folder,
        })
        print(f"[HarnessAgent] Initialized ({HARNESS_VERSION}, mode={self.llm_client.mode}, "
              f"thinking={getattr(args, 'LLM_thinking', 'disabled')}, process {self.process_id})")

    # ------------------------------------------------------------------
    # legacy log files (same names/format as ChatGPTAgent)
    # ------------------------------------------------------------------

    def _save(self, data, filename):
        with open(os.path.join(self.game_folder, filename), "a") as f:
            json.dump(data, f)
            f.write("\n")

    # ------------------------------------------------------------------

    def _get_next_action(self):
        if self.action_queue:
            action_id, slot_source, tick_id = self.action_queue.popleft()
        else:
            action_id, slot_source, tick_id = self.empty_action_idx, "queue_empty", None
        self.telemetry.on_dequeue(action_id, slot_source, tick_id)
        return action_id

    def _record_outcome(self, executed, failures):
        failed_names = set()
        for f in failures if isinstance(failures, list) else []:
            name, reason = parse_failure(f)
            self._failed[(name or "?", reason)] += 1
            if name:
                failed_names.add(name)
        if isinstance(executed, str) and executed != EMPTY_NAME and executed not in failed_names:
            self._executed[executed] += 1

    def _outcome_lines(self):
        lines = [f"{name}: done x{n}" for name, n in sorted(self._executed.items())]
        lines += [f"{name}: FAILED x{n} ({reason})" for (name, reason), n in
                  sorted(self._failed.items(), key=lambda kv: -kv[1])]
        return lines + self._carry_lines

    def _event(self, kind, **payload):
        logger = getattr(self.telemetry, "logger", None)
        if self.telemetry.enabled and logger is not None:
            try:
                logger.log_event(kind, **payload)
                logger.summary_extra.update(skills_loaded=list(self.skills_loaded),
                                            active_skill=self.active_skill, army_goal_final=self.army_goal)
            except Exception as e:
                print(f"[telemetry] event {kind} ignored: {e}")

    def _decide(self, raw, observation=None):
        state = parse_state(raw, observation)
        for name, n in list(state.enemy_units.items()) + list(state.enemy_structures.items()):
            self.enemy_seen[name] = max(self.enemy_seen.get(name, 0), n)
        mask = legal_actions(state, self.bot_rules)
        legal_ids = [i for i, reason in mask.items() if reason is None]
        units = dict(state.units)
        user = build_user_message(
            render_state(state), diff_states(self.prev_state, state), self._outcome_lines(), legal_ids,
            self.plan, enemy_seen=self.enemy_seen,
            goal_lines=goal_progress_lines(self.army_goal, units, state.units_pending, state.supply_used),
            active_skill=(self.active_skill or "") if self.skills else None)
        self._save({"L1_summaries": [[generate_summarize_L1(raw)]], "harness_user_message": user},
                   "combined_input.json")

        messages = [{"role": "system", "content": self.system_prompt}]
        skill = self.skills.get(self.active_skill) if self.active_skill else None
        skill_msg = active_skill_message(self.active_skill, skill.body if skill else None)
        if skill_msg:
            messages.append(skill_msg)
        messages.append({"role": "user", "content": user})

        self.telemetry.before_llm()
        decision, tier, payload, problems, loaded = self.llm_client.decide(messages)
        for name in loaded:
            self.skills_loaded.append(name)
            self.active_skill = name
            self._event("skill_loaded", skill=name, game_time=state.game_time, step=self.current_step,
                        enemy_seen=dict(self.enemy_seen))
            print(f"[HarnessAgent] loaded skill {name}")
        if decision is not None and decision.army_goal is not None and decision.army_goal != self.army_goal:
            self.army_goal = decision.army_goal
            self._event("army_goal_set", goal=self.army_goal, game_time=state.game_time,
                        skill=self.active_skill)
        raw_text = payload if isinstance(payload, str) else json.dumps(payload)
        self._save([raw_text or ""], "commander.json")
        if problems:
            print(f"[HarnessAgent] decision problems: {problems}")

        notes = []
        if decision is None:
            proposed, illegal, parse_tier = [], [], "none"
            notes.append("your last reply was not a valid decision; reply with the decision JSON only")
        else:
            parse_tier = tier
            proposed = [a for a in decision.action_ids if mask.get(a) is None]
            illegal = [a for a in decision.action_ids if mask.get(a) is not None]
            for a in illegal:
                notes.append(f"discarded {a} {PROTOSS_ACTIONS[a].name}: not legal ({mask[a]})")
            if decision.plan:
                self.plan = decision.plan
        print(f"[HarnessAgent] tier={parse_tier} proposed={proposed} illegal={illegal} plan={self.plan!r}")

        # Same downstream as ChatGPTAgent: resource filter, then mix_actions padding.
        ids = list(proposed) if decision is not None else [self.empty_action_idx]
        resources = parse_resources(raw)
        if resources is not None and ids:
            kept, rejections = filter_by_resources(ids, self._id_to_name, resources, self.empty_action_idx)
            if rejections:
                self._save({"step": self.current_step, "resources": resources,
                            "rejections": [{"action": n, "reason": r} for n, r in rejections]},
                           "resource_rejections.json")
                notes += [line.replace("Action failed", "dropped before running")
                          for line in format_rejections_for_actor(rejections)]
        else:
            kept = list(ids)

        sources = source_labels(proposed if decision is not None else ids, kept, parse_tier, self.empty_action_idx)
        slots = mix_with_sources(list(kept), sources, self.action_window, self.action_mix_rate,
                                 self.empty_action_idx)
        tick_id = self.telemetry.on_tick(raw_text, parse_tier, proposed, kept, slots, n_illegal=len(illegal))
        self.action_queue.extend((aid, src, tick_id) for aid, src in slots)

        self.prev_state = state
        self._executed.clear()
        self._failed.clear()
        self._carry_lines = notes
        return self.plan

    def action(self, observation):
        raw = observation['information']
        executed = observation['action_executed']
        failures = observation['action_failures']

        self.telemetry.on_observation(observation)
        self._save(raw, "raw_observation.json")
        self._save(executed, "action_executed.json")
        self._save(failures, "action_failures.json")
        self._record_outcome(executed, failures)

        if not isinstance(raw, dict):
            return self._get_next_action()
        self._save(generate_summarize_L1(raw), "L1_observation.json")

        command, command_flag = None, False
        if self.current_step % self.action_interval == 0:
            command = self._decide(raw, observation) or ""
            command_flag = True

        self.current_step += 1
        return self._get_next_action(), command, command_flag
