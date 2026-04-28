import ast
import json
import os
import numpy as np

from sc2_rl_agent.starcraftenv_test.summarize.L1_summarize import generate_summarize_L1
from sc2_rl_agent.starcraftenv_test.summarize.gpt_test.L2_summarize import L2_summary
from collections import deque
from sc2_rl_agent.starcraftenv_test.utils.action_extractor import *
from sc2_rl_agent.starcraftenv_test.utils.action_vector_test import ActionDBManager
from sc2_rl_agent.starcraftenv_test.utils.resource_filter import (
    parse_resources, filter_by_resources, format_rejections_for_actor,
)


class ChatGPTAgent:
    """
    ChatGPTAgent
    接入大语言模型

    Augmented with deterministic resource-affordability filtering. Before any
    action batch is queued, we drop actions the bot can't currently afford
    (insufficient minerals/gas/supply) and replace them with EMPTY ACTION.
    The rejections are also fed back into the actor's next prompt as
    'Action failed: ..., Reason: ...' lines so the actor learns and stops
    proposing them.

    The earlier two-tier Commander LLM design was removed — it fought the
    actor rather than helping it. Replaced with simple resource math.
    """

    def __init__(self, model_name, api_key, api_base, system_prompt, example_prompt, temperature, args,
                 action_description,
                 raw_observation="raw_observation.json", L1_observation_file_name="L1_observations.json",
                 commander_file_name='commander.json',
                 action_interval=10, chunk_window=5, action_window=10, action_mix_rate=0.5,
                 last_k=5, prompt_type='v2'):
        self.args = args
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.raw_observation_file_name = raw_observation
        self.L1_observation_file_name = L1_observation_file_name
        self.commander_file_name = commander_file_name
        self.raw_observations = []
        self.L1_observations = []
        self.commanders = []
        self.action_interval = action_interval
        self.current_step = 0
        self.example_prompt = example_prompt
        self.action_queue = deque()
        self.summary_queue = deque()
        self.executed_actions_queue = deque()
        self.failed_actions_queue = deque()
        self.chunk_window = chunk_window
        self.action_window = action_window
        self.action_mix_rate = action_mix_rate
        self.last_k = last_k
        self.action_description = action_description
        self.action_dict = self.action_description.action_descriptions
        self.temp_command = "temp command"
        self.current_time = args.current_time
        self.game_info = self.generate_game_info()
        self.process_id = args.process_id
        self.empty_action_idx = self.get_empty_action_idx()
        self.L2 = L2_summary(LLMapi_base=self.api_base, LLMapi_key=self.api_key, model_name=self.model_name,
                             temperature=self.temperature, system_prompt=self.system_prompt,
                             example_prompt=self.example_prompt, chunk_window=self.chunk_window,
                             prompt_type=prompt_type)
        self.get_opposite_bot_name()
        self.base_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log', 'chatgpt_log')
        self.action_db_manager = self.init_action_db()
        self.game_folder = os.path.join(self.base_save_dir, f"game_{self.current_time}_{self.process_id}")
        if not os.path.exists(self.game_folder):
            os.makedirs(self.game_folder)

        # Cache action_id -> action_name map for fast lookup during filtering.
        self._id_to_name = {}
        for cat, actions in self.action_dict.items():
            for aid, aname in actions.items():
                self._id_to_name[aid] = aname

        # Track resource-rejected actions across ticks so we can feed them
        # back to the actor as 'Action failed: ...' lines on the next prompt.
        self._recent_resource_rejections: list = []

        print(f"[ChatGPTAgent] Initialized. Resource filter active "
              f"(process {self.process_id})")

    def init_action_db(self):
        action_vdb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "actionvdb", "action_vdb")
        self.action_db = ActionDBManager(db_path=action_vdb_path)
        if self.args.player_race == "Protoss":
            self.action_db.initialize_collection("protoss_actions")
            return self.action_db
        elif self.args.player_race == "Zerg":
            self.action_db.initialize_collection("zerg_actions")
            return self.action_db
        else:
            raise ValueError("Not support Race")

    # -----------------------------------------------------------------------
    # Resource filter integration
    # -----------------------------------------------------------------------

    def _apply_resource_filter(self, action_ids, raw_observation):
        """
        Walk the actor's proposed action_ids in order, dropping any that we
        can't currently afford. Replaces unaffordable actions with EMPTY
        ACTION so the queue length stays consistent with mix_actions.

        Currently only enabled for Protoss — the cost table covers Protoss
        vocab. Zerg/Terran would need their own cost tables.
        """
        if self.args.player_race != "Protoss":
            return action_ids

        resources = parse_resources(raw_observation)
        if resources is None:
            # Couldn't read resources — be permissive rather than blocking
            # everything. The env will still reject unaffordable actions
            # naturally on execution.
            return action_ids

        kept, rejections = filter_by_resources(
            action_ids, self._id_to_name, resources, self.empty_action_idx,
        )
        if rejections:
            print(f"[ResourceFilter] dropped {len(rejections)}/{len(action_ids)} "
                  f"unaffordable actions: {[r[0] for r in rejections]} "
                  f"(have {resources['minerals']}m, {resources['gas']}g, "
                  f"{resources['supply_left']} free supply)")
            self._save_data_to_file(
                {
                    "step": self.current_step,
                    "resources": resources,
                    "rejections": [{"action": n, "reason": r} for n, r in rejections],
                },
                "resource_rejections.json",
            )
            self._recent_resource_rejections = format_rejections_for_actor(rejections)
        else:
            self._recent_resource_rejections = []
        return kept

    # -----------------------------------------------------------------------

    def preprocess_actions(self):
        """
        Build the executed/failed lists fed into the actor's next prompt.

        Now also includes resource-rejection notes from the previous tick,
        so the actor sees "Action failed: BUILD GATEWAY, Reason: need 150
        minerals, have 95" in the failed_actions section and learns.
        """
        executed_actions = [action for action in self.executed_actions_queue if action != "EMPTY ACTION"]
        failed_actions_list = self.failed_actions_queue
        failed_actions_structured = []
        for failure in failed_actions_list:
            for f in failure:
                failed_actions_structured.append(f)
        # Append resource-filter rejections from the prior tick.
        failed_actions_structured.extend(self._recent_resource_rejections)
        return executed_actions, failed_actions_structured

    def _save_data_to_file(self, data, filename):
        full_path = os.path.join(self.game_folder, filename)
        with open(full_path, "a") as file:
            json.dump(data, file)
            file.write("\n")

    def _save_raw_observation_to_file(self, raw_observation):
        self._save_data_to_file(raw_observation, "raw_observation.json")

    def _save_action_executed_to_file(self, action_executed):
        self._save_data_to_file(action_executed, "action_executed.json")

    def _save_action_failures_to_file(self, action_failures):
        self._save_data_to_file(action_failures, "action_failures.json")

    def _save_L1_observation_to_file(self, L1_observation):
        self._save_data_to_file(L1_observation, "L1_observation.json")

    def _save_commander_to_file(self, commander):
        """Saves the actor's command output. Filename kept for backward
        compatibility — this is the actor's command, not a strategic
        Commander module (which has been removed)."""
        self._save_data_to_file(commander, "commander.json")

    def _save_combined_input_to_file(self, combined_input):
        self._save_data_to_file(combined_input, "combined_input.json")

    def get_empty_action_idx(self):
        flat_dict = {}
        for key, value in self.action_dict.items():
            for inner_key, inner_value in value.items():
                flat_dict[inner_key] = inner_value
        empty_action_idx = len(flat_dict) - 1
        return empty_action_idx

    def get_opposite_bot_name(self):
        if self.args.opposite_type == 'build_in':
            self.opposite_name = self.args.difficulty
        elif self.args.opposite_type == 'rule':
            self.opposite_name = self.args.opposite_bot
        else:
            raise ValueError("opposite_type must be build_in or rule")

    def generate_game_info(self):
        if not hasattr(self, 'opposite_name'):
            self.get_opposite_bot_name()
        game_info = (f"Map_{self.args.map_pool[self.args.map_idx]}_"
                     f"Player_race_{self.args.player_race}_vs_{self.args.opposite_race}_"
                     f"opposite_type_{self.args.opposite_type}_{self.opposite_name}")
        return game_info

    def _get_next_action(self):
        if self.action_queue:
            return self.action_queue.popleft()
        return self.empty_action_idx

    def extract_actions_from_command(self, command):
        if isinstance(command, list):
            command = " ".join(command)
        self.action_extractor = ActionExtractor(self.action_dict)
        empty_idx = self.action_description.empty_action_id
        action_ids, valid_actions = extract_actions_from_command(
            command, action_extractor=self.action_extractor,
            empty_idx=empty_idx, action_db_manager=self.action_db_manager,
        )
        return action_ids, valid_actions

    def action(self, observation):
        """
        Generate the next action.

        Flow:
          1. Save raw observation, executed/failed actions to disk.
          2. If raw_observation is a dict, generate L1 summary; else dequeue.
          3. Every action_interval steps:
               a. Build combined_input from L1 summaries + executed +
                  failed. Failed now includes resource rejections from the
                  prior tick, so the actor sees "Action failed: ... cannot
                  afford" feedback and learns.
               b. Query L2 for a command.
               c. Extract action_ids from the command.
               d. APPLY RESOURCE FILTER — drop unaffordable actions.
               e. Mix with empties, queue.
        """
        player_race = observation['player_race']
        opposite_race = observation['opposite_race']
        map_name = observation['map_name']
        raw_observation = observation['information']
        action_executed = observation['action_executed']
        action_failures = observation['action_failures']

        self.executed_actions_queue.append(action_executed)
        self.failed_actions_queue.append(action_failures)

        self._save_raw_observation_to_file(raw_observation)
        self._save_action_executed_to_file(action_executed)
        self._save_action_failures_to_file(action_failures)

        if isinstance(raw_observation, dict):
            L1_observation = generate_summarize_L1(raw_observation)
            self._save_L1_observation_to_file(L1_observation)
        else:
            return self._get_next_action()

        self.summary_queue.append(L1_observation)

        command = None
        command_flag = False

        if self.current_step % self.action_interval == 0 and self.summary_queue:
            summaries = [list(self.summary_queue)]
            last_k_L1_summaries = self.L2.get_latest_k_messages(summaries, self.last_k)
            executed, failed = self.preprocess_actions()

            combined_input = {
                'L1_summaries': last_k_L1_summaries,
                'executed_actions': executed,
                'failed_actions': failed,
            }
            self._save_combined_input_to_file(combined_input)

            L2_summaries = self.L2.query(combined_input)
            command = L2_summaries
            self._save_commander_to_file(command)
            self.temp_command = command
            print('command:', command)

            action_ids, action_values = self.extract_actions_from_command(command)
            print(f"[debug] extracted {len(action_ids)} actions: {action_ids}")

            # ---- Resource filter: drop unaffordable actions ----
            action_ids = self._apply_resource_filter(action_ids, raw_observation)
            # ----------------------------------------------------

            mixed_actions = self.mix_actions(action_ids)
            self.action_queue.extend(mixed_actions)
            print("action_queue:", self.action_queue)

            self.summary_queue.clear()
            self.executed_actions_queue.clear()
            self.failed_actions_queue.clear()
            command_flag = True

        self.current_step += 1
        action = self._get_next_action()
        return action, command, command_flag

    def mix_actions(self, real_actions):
        """
        混合真实动作和空动作。Note: real_actions may already contain EMPTY
        ACTION entries from the resource filter; those just become more
        empties in the mixed sequence.
        """
        empty_action = self.empty_action_idx
        mixed_actions = []
        num_real_actions = int(self.action_window * self.action_mix_rate)
        if num_real_actions > len(real_actions):
            num_real_actions = len(real_actions)
        real_action_indices = np.linspace(start=0, stop=self.action_window - 1, num=num_real_actions, dtype=int)
        for i in range(self.action_window):
            if i in real_action_indices and real_actions:
                mixed_actions.append(real_actions.pop(0))
            else:
                mixed_actions.append(empty_action)
        return mixed_actions