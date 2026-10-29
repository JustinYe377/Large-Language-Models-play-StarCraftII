# Must run before anything imports sc2: selects headless Linux vs Windows SC2.
from sc2_rl_agent.starcraftenv_test.sc2_runtime import (
    apply_sc2_runtime_args, add_sc2_runtime_args, add_game_run_args, check_install,
)
apply_sc2_runtime_args()

from datetime import datetime

import argparse

from sc2_rl_agent.starcraftenv_test.utils.action_info import ActionDescriptions
from sc2_rl_agent.starcraftenv_test.config.config import LADDER_MAP_2023, DIFFICULTY_LEVELS, AI_BUILD_LEVELS
from sc2_rl_agent.starcraftenv_test.env.starcraft_env import StarCraftEnvSelector
from sc2_rl_agent.starcraftenv_test.agent.chatgpt_agent import ChatGPTAgent
from sc2_rl_agent.starcraftenv_test.agent.random_agent import RandomAgent
from sc2_rl_agent.starcraftenv_test.agent.real_time_agent import RealTimeAgent
from sc2_rl_agent.starcraftenv_test.agent.llama2_agent import Llama2_Agent as Llama2Agent
from sc2_rl_agent.starcraftenv_test.prompt.prompt import *
from sc2_rl_agent.starcraftenv_test.telemetry.cli import add_telemetry_args

"""
地图池

    laddermap_2023 = ['Altitude LE', 'Ancient Cistern LE', 'Babylon LE', 'Dragon Scales LE', 'Gresvan LE',
                      'Neohumanity LE', 'Royal Blood LE']

build in ai 难度

'Difficulty.values',
  full_name='SC2APIProtocol.Difficulty',
  values=['VeryEasy', 'Easy', 'Medium', 'MediumHard', 'Hard', 'Harder''VeryHard''CheatVision''CheatMoney','CheatInsane']
  terran_bot = [marine_marauder_Bot,marine_tank_Bot,tank_heller_thor_Bot]
  protoss_bot =[WarpGateBot]
  zerg_bot = [hydra_ling_bane_bot,roach_ling_baneling_bot,roach_hydra_bot]
  
"""


def agent_test(agent, env):
    observation, _ = env.reset()  # Get initial observation
    done = False
    result = None

    while not done:
        action = agent.action(observation)
        observation, reward, done, result, info = env.step(action)

        if done:
            break
    return result, observation


def initialize_and_test_agent(args):
    sc2_info = check_install(args.map_pool[args.map_idx])
    print(f"[test_the_env] SC2 {sc2_info}")

    # 初始化环境
    selector = StarCraftEnvSelector(args)
    env = selector.create_env()

    # 获取行动描述
    action_description = ActionDescriptions(env.player_race)
    action_dict = action_description.action_descriptions
    print("action_dict", action_dict)

    # 生成提示
    K = 5
    sc2prompt = StarCraftIIPrompt_V2(race=args.player_race, K=str(K), action_dict=action_dict,game_style="aggressive")
    system_prompt, example_input_prompt, example_output_prompt = sc2prompt.generate_prompts()
    example_prompt = [example_input_prompt.format(K_1=K - 1), example_output_prompt]
    # Recorded in the telemetry run config.
    args.prompt_class = type(sc2prompt).__name__
    args.prompt_K = K

    # 创建并测试agent
    if args.agent_type == 'random':
        agent = RandomAgent(args.LLM_model_name, args.LLM_api_key, args.LLM_api_base,
                            system_prompt, example_prompt, args.LLM_temperature,
                            args, action_description)
        agent_test(agent, env)
    elif args.agent_type in ('gpt', 'chatgpt', 'deepseek'):
        # All three aliases route to ChatGPTAgent. DeepSeek is OpenAI-compatible,
        # so the same agent class works as long as --LLM_api_base points at it.
        agent = ChatGPTAgent(args.LLM_model_name, args.LLM_api_key, args.LLM_api_base,
                             system_prompt, example_prompt, args.LLM_temperature,
                             args, action_description)
        # Always close the telemetry run, including on Ctrl+C or a crash.
        result, final_observation = "crash", None
        try:
            result, final_observation = agent_test(agent, env)
        except KeyboardInterrupt:
            result = "aborted"
            raise
        finally:
            agent.telemetry.finish(result, final_observation)
    elif args.agent_type == 'llama2':
        agent = Llama2Agent(args.LLM_model_name, args.LLM_api_key, args.LLM_api_base,
                            system_prompt, example_prompt, args.LLM_temperature,
                            args, action_description)
        agent_test(agent, env)


    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")


def real_time_test(args):
    # 初始化环境
    selector = StarCraftEnvSelector(args)
    env = selector.create_env()

    # 获取行动描述
    action_description = ActionDescriptions(env.player_race)
    action_dict = action_description.action_descriptions
    print("action_dict", action_dict)

    # 生成提示
    sc2prompt = StarCraftIIPrompt_realtime(race=args.player_race, K="10", action_dict=action_dict)
    system_prompt, example_input_prompt, example_output_prompt = sc2prompt.generate_prompts()
    example_prompt = [example_input_prompt.format(K_1=9), example_output_prompt]

    # 创建并测试agent
    if args.agent_type == 'random':
        agent = RandomAgent(args.LLM_model_name, args.LLM_api_key, args.LLM_api_base,
                            system_prompt, example_prompt, args.LLM_temperature,
                            args, action_description)
        agent_test(agent, env)
    elif args.agent_type == 'chatgpt':
        agent = RealTimeAgent(args.LLM_model_name, args.LLM_api_key, args.LLM_api_base,
                              system_prompt, example_prompt, args.LLM_temperature,
                              args, action_description)
        agent_test(agent, env)
    elif args.agent_type == 'llama2':
        agent = Llama2Agent(args.LLM_model_name, args.LLM_api_key, args.LLM_api_base,
                            system_prompt, example_prompt, args.LLM_temperature,
                            args, action_description)
        agent_test(agent, env)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StarCraft II environment testing tool.')

    parser.add_argument('--num_agents', type=str, default='single',
                        help='Choose "single" for single bot or "two" for two agents.')

    parser.add_argument('--env_type', type=str, default='text', help='Environment type.')

    parser.add_argument('--map_pool', type=list, default=LADDER_MAP_2023, help='List of maps for the game.')

    parser.add_argument('--map_idx', type=int, default=1, help='Index of the map to use from the map pool.')

    parser.add_argument('--player_race', type=str, default='Protoss',
                        help='Player race. Use "Protoss" or "Zerg". Only valid for single bot.')

    parser.add_argument('--opposite_race', type=str, default='Zerg',
                        help='Opponent race. Use "Protoss" or "Zerg". Only valid for single bot.')

    parser.add_argument('--opposite_type', type=str, default='build_in',
                        help='Opponent bot type. Use "rule" or "build_in". "rule" means bot AI designed by makers, "build_in" means official Blizzard AI. Only valid for single bot.')

    parser.add_argument('--opposite_bot', type=str, default='hydra_ling_bane_bot',
                        help='Opponent bot type when playing against rule AI. Only valid for single bot.')

    parser.add_argument('--difficulty', type=str, default=DIFFICULTY_LEVELS[2],
                        help='Game difficulty level when playing against build-in AI. Only valid for single bot.')

    parser.add_argument('--ai_build', type=str, default=AI_BUILD_LEVELS[0], help='ai build level')

    parser.add_argument('--player1_race', type=str, default='Zerg',
                        help='Player 1 race. Use "Protoss" or "Zerg". Only valid for two agents.')

    parser.add_argument('--player2_race', type=str, default='Protoss',
                        help='Player 2 race. Use "Protoss" or "Zerg". Only valid for two agents.')

    parser.add_argument('--process_id', type=str, default='-1', help='-1 means not multiprocess worker'
                                                                     '0-100 means multiprocess id.')
    parser.add_argument('--current_time', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help='Current time. Default is current system time.')
    parser.add_argument('--agent_type', type=str, default="gpt",
                        help='Agent type. Use "random","gpt","chatgpt","deepseek","llama2","glm2"')

    parser.add_argument('--LLM_model_name', type=str, default="deepseek-v4-flash",
                        help="e.g. deepseek-v4-flash, deepseek-chat, gpt-3.5-turbo-16k")
    parser.add_argument('--LLM_temperature', type=float, default=0.3)

    parser.add_argument('--LLM_api_key', type=str, default="Your-api-key")
    parser.add_argument('--LLM_api_base', type=str, default="https://api.deepseek.com/v1")

    # NOTE: argparse 'type=bool' is broken — bool("False") is True. Use mutually-
    # exclusive store_true / store_false flags instead. Default is real_time=False.
    real_time_group = parser.add_mutually_exclusive_group()
    real_time_group.add_argument('--real_time', dest='real_time', action='store_true',
                                 help='Run SC2 in real-time mode (slow, human-watchable).')
    real_time_group.add_argument('--no_real_time', dest='real_time', action='store_false',
                                 help='Run SC2 in non-realtime / fast-forward mode (default).')
    parser.set_defaults(real_time=False)

    add_telemetry_args(parser)
    add_sc2_runtime_args(parser)
    add_game_run_args(parser)

    # Commander strategy: 'macro' (default) or 'rush' (4-Gate Stalker all-in).
    parser.add_argument('--strategy', type=str, default='macro',
                        choices=['macro', 'rush'],
                        help="Commander strategy. 'macro' = standard nexus-first into "
                             "tech (default). 'rush' = 4-Gate Stalker all-in.")

    args = parser.parse_args()

    print(f"[test_the_env] agent_type={args.agent_type} model={args.LLM_model_name} "
          f"strategy={args.strategy} real_time={args.real_time} difficulty={args.difficulty}")

    if args.real_time:
        real_time_test(args)
    else:
        initialize_and_test_agent(args)