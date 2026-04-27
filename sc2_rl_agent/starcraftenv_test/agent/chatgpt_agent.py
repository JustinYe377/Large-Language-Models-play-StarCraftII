import ast
import json
import os
import time
import numpy as np
import openai
from sc2_rl_agent.starcraftenv_test.summarize.L1_summarize import generate_summarize_L1
from sc2_rl_agent.starcraftenv_test.summarize.gpt_test.L2_summarize import L2_summary
from collections import deque
from sc2_rl_agent.starcraftenv_test.utils.action_extractor import *
from sc2_rl_agent.starcraftenv_test.utils.action_vector_test import ActionDBManager
from sc2_rl_agent.starcraftenv_test.commander import (
    Commander, PROTOSS_ACTION_VOCAB, PROTOSS_NEVER_FORBID,
)


class ChatGPTAgent:
    """
    ChatGPTAgent
    接入大语言模型

    Now augmented with a strategic-tier Commander (see commander.py) that
    fires every ~60 game-seconds and constrains the actor's action space
    to prevent commitment-failure bugs (e.g. spending while saving for a
    second Nexus).
    """

    def __init__(self, model_name, api_key, api_base, system_prompt, example_prompt, temperature, args,
                 action_description,
                 raw_observation="raw_observation.json", L1_observation_file_name="L1_observations.json",
                 commander_file_name='commander.json',
                 action_interval=10, chunk_window=5, action_window=10, action_mix_rate=0.5,
                 last_k=5, prompt_type='v3'):
        self.args = args
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.system_prompt = system_prompt  # 系统提示
        self.temperature = temperature  # 生成文本的多样性
        self.raw_observation_file_name = raw_observation  # 保存原始观察的文件名
        self.L1_observation_file_name = L1_observation_file_name  # 保存L1观察的文件名
        self.commander_file_name = commander_file_name  # 保存命令的文件名
        self.raw_observations = []  # List to store raw_observations
        self.L1_observations = []  # List to store L1_observations
        self.commanders = []  # List to store commanders
        self.action_interval = action_interval  # 每隔几个step执行一个真实的动作
        self.current_step = 0  # 当前步数
        self.example_prompt = example_prompt  # 例子输入
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
        self.temp_command = "temp command"  # 用于保存临时的command
        self.current_time = args.current_time  # 获取当前时间
        self.game_info = self.generate_game_info()  # 生成游戏信息
        self.process_id = args.process_id
        self.empty_action_idx = self.get_empty_action_idx()
        self.L2 = L2_summary(LLMapi_base=self.api_base, LLMapi_key=self.api_key, model_name=self.model_name,
                             temperature=self.temperature, system_prompt=self.system_prompt,
                             example_prompt=self.example_prompt, chunk_window=self.chunk_window,
                             prompt_type=prompt_type)
        self.get_opposite_bot_name()
        self.base_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'log', 'chatgpt_log')
        self.action_db_manager = self.init_action_db()
        # 为每一局游戏创建一个独立的文件夹，这里使用当前时间作为唯一标识符
        self.game_folder = os.path.join(self.base_save_dir, f"game_{self.current_time}_{self.process_id}")

        # 如果目录不存在，创建它
        if not os.path.exists(self.game_folder):
            os.makedirs(self.game_folder)

        # Cache action_id -> action_name mapping for fast forbidden-action lookup.
        self._id_to_name = {}
        for cat, actions in self.action_dict.items():
            for aid, aname in actions.items():
                self._id_to_name[aid] = aname

        # ---- Commander setup -----------------------------------------------
        # The Commander is the strategic-tier LLM that fires every ~60 game
        # seconds. Currently only Protoss is supported (vocab is hard-coded
        # in commander.py). For Zerg/Terran the commander is disabled and
        # the actor falls back to its prior behavior.
        if self.args.player_race == "Protoss":
            self.commander = Commander(
                llm_call=self._commander_llm_call,
                tick_interval_game_seconds=60.0,
                known_actions=PROTOSS_ACTION_VOCAB,
                never_forbid=list(PROTOSS_NEVER_FORBID),
            )
            # Skip the t=0 tick. At game start there's no useful state for
            # the commander to reason about — 50 minerals, 12 probes, no
            # scouting info. First real tick fires at ~60s game time, when
            # the bot is actually making meaningful decisions.
            self.commander.state.last_tick_game_seconds = 0.0
            print(f"[ChatGPTAgent] Commander enabled (process {self.process_id}); "
                  f"first tick scheduled around game-time 60s")
        else:
            self.commander = None
            print(f"[ChatGPTAgent] Commander disabled — race={self.args.player_race}")
        # --------------------------------------------------------------------

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
    # Commander integration helpers
    # -----------------------------------------------------------------------

    def _commander_llm_call(self, system: str, user: str) -> str:
        """
        Bridge from Commander's callable interface to the openai client.
        Reuses the same api_base/api_key/model as the actor — that means
        DeepSeek (or whatever you configured via --LLM_api_base).

        Retries on:
          * Empty content (DeepSeek occasionally returns "" on the first
            request of a session, especially with V4 models).
          * Network/transient errors.
        Each retry uses exponential backoff. After 3 failed attempts we
        raise; Commander._tick will catch the exception and keep prior
        state, so a failed commander call never crashes the game.
        """
        openai.api_key = self.api_key
        openai.api_base = self.api_base

        last_err = None
        for attempt in range(3):
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=min(0.3, self.temperature),  # commander wants stable JSON
                    max_tokens=600,                           # V4 needs headroom
                    request_timeout=30,                       # don't hang forever
                )
                content = resp["choices"][0]["message"]["content"] or ""
                if content.strip():
                    return content
                last_err = "empty content from LLM"
                print(f"[Commander] attempt {attempt + 1}/3: empty response, retrying")
            except Exception as e:
                last_err = repr(e)
                print(f"[Commander] attempt {attempt + 1}/3: {last_err}")

            # Exponential backoff: 1s, 2s, 4s.
            time.sleep(2 ** attempt)

        raise RuntimeError(
            f"commander LLM produced no usable response after 3 attempts: {last_err}"
        )

    def _estimate_game_seconds(self, raw_observation):
        """
        Extract real game-time seconds from raw_observation.

        TextStarCraft2 stores game time inside raw_observation['resource'] as
        a STRINGIFIED Python dict whose 'game_time' field is "MM:SS". Example:
            raw_observation['resource'] == "{'game_time': '03:45', ...}"

        Notes:
          * It's str(dict), not JSON, so we use ast.literal_eval (json.loads
            chokes on single quotes).
          * If parsing fails we fall back to a step-based estimate so the
            commander still ticks at a reasonable cadence rather than crashing.
        """
        if isinstance(raw_observation, dict):
            resource = raw_observation.get('resource')
            if isinstance(resource, str):
                try:
                    parsed = ast.literal_eval(resource)
                    gt = parsed.get('game_time') if isinstance(parsed, dict) else None
                    if isinstance(gt, str) and ':' in gt:
                        mm, ss = gt.split(':', 1)
                        return int(mm) * 60 + int(ss)
                except (ValueError, SyntaxError, AttributeError):
                    pass
            elif isinstance(resource, dict):
                # In case the upstream ever stops stringifying the dict.
                gt = resource.get('game_time')
                if isinstance(gt, str) and ':' in gt:
                    mm, ss = gt.split(':', 1)
                    try:
                        return int(mm) * 60 + int(ss)
                    except ValueError:
                        pass
        # Fallback: ~0.357 game-sec per agent step at default step_mul=8.
        # Only hit if the resource field is missing or malformed.
        return self.current_step * (8 / 22.4)

    def _filter_actions(self, action_ids):
        """
        Replace any action_id whose name is rejected by the Commander with
        the EMPTY ACTION id.

        Uses commander.is_allowed(name), which handles three layers in order:
          1. Allowlist mode (saving_for is critical) — only allowlisted actions pass.
          2. Tech commitment — actions on a non-committed tech path are blocked.
          3. Plain blacklist — explicit forbidden_actions are blocked.
        """
        if self.commander is None:
            return action_ids
        # Skip work if there are no constraints active at all.
        s = self.commander.state
        if not s.allowed_actions and s.committed_tech_path == "none" and not s.forbidden_actions:
            return action_ids

        kept = []
        dropped = []
        for aid in action_ids:
            aname = self._id_to_name.get(aid)
            if aname is None:
                kept.append(aid)  # unknown id, leave alone
                continue
            if self.commander.is_allowed(aname):
                kept.append(aid)
            else:
                dropped.append(aname)
                kept.append(self.empty_action_idx)
        if dropped:
            print(f"[Commander] hard-filtered actions from batch: {dropped}")
        return kept

    # Back-compat alias — old code path may still call the old name.
    _filter_forbidden_action_ids = _filter_actions

    # -----------------------------------------------------------------------

    def preprocess_actions(self):
        # Convert executed actions to a list without 'EMPTY ACTION'
        executed_actions = [action for action in self.executed_actions_queue if action != "EMPTY ACTION"]

        # Convert failed actions to a structured format
        failed_actions_list = self.failed_actions_queue
        failed_actions_structured = []
        for failure in failed_actions_list:
            for f in failure:
                failed_actions_structured.append(f)

        return executed_actions, failed_actions_structured

    def _save_data_to_file(self, data, filename):
        """
        通用的数据保存方法
        :param data: 要保存的数据
        :param filename: 保存的文件名
        :return: None
        """
        full_path = os.path.join(self.game_folder, filename)
        with open(full_path, "a") as file:
            json.dump(data, file)
            file.write("\n")

    def _save_raw_observation_to_file(self, raw_observation):
        """保存观测信息"""
        filename = "raw_observation.json"
        self._save_data_to_file(raw_observation, filename)

    def _save_action_executed_to_file(self, action_executed):
        """保存已执行的动作信息"""
        filename = "action_executed.json"
        self._save_data_to_file(action_executed, filename)

    def _save_action_failures_to_file(self, action_failures):
        """保存失败的动作信息"""
        filename = "action_failures.json"
        self._save_data_to_file(action_failures, filename)

    def _save_L1_observation_to_file(self, L1_observation):
        """保存L1 summarize后的信息"""
        filename = "L1_observation.json"
        self._save_data_to_file(L1_observation, filename)

    def _save_commander_to_file(self, commander):
        """保存GPT输出的command信息 (NOTE: this is the actor's command,
        not the strategic Commander module — naming preserved for back-compat)."""
        filename = "commander.json"
        self._save_data_to_file(commander, filename)

    def _save_combined_input_to_file(self, combined_input):
        """保存LLM决策时的输入"""
        filename = "combined_input.json"
        self._save_data_to_file(combined_input, filename)

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
        """
        生成游戏信息
        地图
        玩家种族
        对手种族
        对手类型
        """
        if not hasattr(self, 'opposite_name'):
            self.get_opposite_bot_name()
        game_info = f"Map_{self.args.map_pool[self.args.map_idx]}_Player_race_{self.args.player_race}_vs_{self.args.opposite_race}_opposite_type_{self.args.opposite_type}_{self.opposite_name}"
        return game_info

    def _get_next_action(self):
        # Check if there are actions in the queue
        if self.action_queue:
            return self.action_queue.popleft()
        else:
            return self.empty_action_idx

    def extract_actions_from_command(self, command):
        if isinstance(command, list):
            command = " ".join(command)
        self.action_extractor = ActionExtractor(self.action_dict)
        empty_idx = self.action_description.empty_action_id
        action_ids, valid_actions = extract_actions_from_command(command, action_extractor=self.action_extractor,
                                                                 empty_idx=empty_idx,
                                                                 action_db_manager=self.action_db_manager)
        return action_ids, valid_actions

    def action(self, observation):
        """
        Generate the next action for the ChatGPT agent.

        Flow with Commander integration:
          1. Save raw observation, executed/failed actions to disk.
          2. If raw_observation is a dict, generate L1 summary and queue it.
          3. Every action_interval steps:
               a. Tick the Commander (if Protoss). It updates its strategic
                  state — saving_for, forbidden_actions, intent.
               b. Pass commander directive into combined_input so L2.query
                  can splice it into the actor's prompt.
               c. Get the actor's command from L2.
               d. Extract action_ids from the command.
               e. Hard-filter: replace any action_id whose name is in
                  commander.forbidden_actions with EMPTY ACTION.
               f. Queue the filtered actions.
        """

        # Extract the raw observation from the list and save it to a file.
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

        # If the raw observation is a dictionary, generate a level 1 summary
        # and save it. Otherwise, return the next action from the queue.
        if isinstance(raw_observation, dict):
            L1_observation = generate_summarize_L1(raw_observation)
            self._save_L1_observation_to_file(L1_observation)
        else:
            return self._get_next_action()

        # Add the new level 1 summary to the queue.
        self.summary_queue.append(L1_observation)

        # Initialize the command and the command flag.
        command = None
        command_flag = False

        # Decision tick — every action_interval steps.
        if self.current_step % self.action_interval == 0 and self.summary_queue:
            summaries = [list(self.summary_queue)]
            last_k_L1_summaries = self.L2.get_latest_k_messages(summaries, self.last_k)
            executed, failed = self.preprocess_actions()

            # ---- Commander tick --------------------------------------------
            game_seconds = self._estimate_game_seconds(raw_observation)
            commander_block = ""
            if self.commander is not None:
                latest_L1 = last_k_L1_summaries[-1] if last_k_L1_summaries else ""
                commander_summary = (
                    f"Game time: {game_seconds:.0f}s\n"
                    f"Latest state summary: {latest_L1}\n"
                    f"Recent executed actions: {executed[-10:] if executed else 'none'}"
                )
                ticked = self.commander.maybe_tick(game_seconds, commander_summary)
                if ticked:
                    s = self.commander.state
                    mode = ("ALLOWLIST" if s.allowed_actions
                            else "TECH+BLACKLIST" if s.committed_tech_path != "none"
                            else "BLACKLIST" if s.forbidden_actions
                            else "FREE")
                    print(f"[Commander] tick at {game_seconds:.0f}s [{mode}]: "
                          f"intent='{s.intent}' "
                          f"saving_for={s.saving_for} "
                          f"tech={s.committed_tech_path} "
                          f"allowed={s.allowed_actions if s.allowed_actions else '(none)'} "
                          f"forbidden={s.forbidden_actions}")
                    self._save_data_to_file(
                        {
                            "game_seconds": game_seconds,
                            "intent": s.intent,
                            "saving_for": s.saving_for,
                            "tech_path": s.committed_tech_path,
                            "tech_committed_at_seconds": s.tech_committed_at_seconds,
                            "allowed_actions": s.allowed_actions,
                            "forbidden_actions": s.forbidden_actions,
                            "raw": s.raw_last_response,
                        },
                        "commander_ticks.json",
                    )
                commander_block = self.commander.get_prompt_block()
            # ----------------------------------------------------------------

            combined_input = {
                'L1_summaries': last_k_L1_summaries,
                'executed_actions': executed,
                'failed_actions': failed,
                # Pass commander directive through. Whether L2.query actually
                # splices it into the prompt depends on L2_summary — see the
                # corresponding patch to L2_summarize.py.
                'commander_directive': commander_block,
            }

            self._save_combined_input_to_file(combined_input)

            # Generate a level 2 summary and get a command.
            L2_summaries = self.L2.query(combined_input)
            command = L2_summaries

            self._save_commander_to_file(command)
            self.temp_command = command
            print('command:', command)

            # Extract the action ids from the command.
            action_ids, action_values = self.extract_actions_from_command(command)

            # ---- Commander hard-lock filter --------------------------------
            # Uses commander.is_allowed() — handles allowlist, tech commitment,
            # and blacklist in one pass. Forbidden actions become EMPTY ACTION.
            action_ids = self._filter_actions(action_ids)
            # ----------------------------------------------------------------

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
        混合真实动作和空动作,用来实现每隔几个step执行一个真实的动作,其余的动作都是空动作,
        实现了动作的稀疏性,缓解了LLM的计算压力,使得LLM和游戏引擎可以较为正常的交互
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