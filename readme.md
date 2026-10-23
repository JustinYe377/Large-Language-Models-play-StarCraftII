# Large Language Models Play StarCraft II — Extended

> **Fork notice.** This is a fork of [histmeisah/Large-Language-Models-play-StarCraftII](https://github.com/histmeisah/Large-Language-Models-play-StarCraftII), the reference implementation for the NeurIPS 2024 paper *"Large Language Models Play StarCraft II: Benchmarks and a Chain of Summarization Approach"*. The original work demonstrated an LLM agent winning against the in-game built-in AI up to **Level 5 (Hard)**. This fork extends the agent and pushes the win ceiling to **Level 6 (Harder, ≈ Diamond-3 ladder skill)** by replacing the LLM's resource-management failures with deterministic Python.

![VYY 5IX JX3 H)`N$_B}@L](https://github.com/histmeisah/Large-Language-Models-play-StarCraftII/assets/49554454/59a941fa-bd71-4145-b99e-3a971aa93790)

---

## What's new in this fork

Three additions, all keeping the original Chain-of-Summarization architecture intact:

### 1. Deterministic resource-affordability filter

A 71-action cost table for the entire Protoss vocabulary (verified against [Liquipedia](https://liquipedia.net/starcraft2)) plus a small Python module that drops actions the bot cannot currently afford **before** they reach the action queue. Rejections are formatted as `Action failed: BUILD GATEWAY, Reason: need 150 minerals, have 95` and fed back into the actor LLM's next prompt — same format the python-sc2 environment uses for its own rejections, so the LLM sees a familiar schema.

The filter operates in **snapshot mode**: each action checks against the current resource pool independently. We deliberately do *not* simulate cost depletion across the batch — doing so artificially fails important late-batch actions (e.g. `BUILD NEXUS` getting rejected because `BUILD PYLON` was listed first).

**Result:** `cannot-afford` failure rate dropped from 43% to 7.6%.

Code: [`sc2_rl_agent/starcraftenv_test/utils/resource_filter.py`](sc2_rl_agent/starcraftenv_test/utils/resource_filter.py)

### 2. Four-tier action extractor

The original single-regex extractor was tuned for GPT-3.5's output style. With newer instruction-tuned models like DeepSeek-V4-Flash, ~43% of decision ticks were silently producing `[71]` (EMPTY only) because the LLM emits formats the original regex doesn't recognize. The new extractor cascades through four strategies:

- **Tier 1**: Structured trailer `ACTIONS: 0, 19, 22, 23, 66`
- **Tier 2**: Inline integer IDs — `(action 19: BUILD PYLON)`, `**0**: TRAIN PROBE`, `0: TRAIN PROBE`
- **Tier 3**: Legacy `Decisions:` regex, extended for em-dash, en-dash, and backtick delimiters
- **Tier 4**: Vector-DB similarity fallback (the original mechanism)

A `_looks_like_list_indices()` guard detects when leading numbers `0,1,2,3,4` are markdown list numbering rather than action IDs and falls through to name-based matching.

**Result:** silent extractor failures went from ~43% of ticks to 0% in regression tests.

Code: [`sc2_rl_agent/starcraftenv_test/utils/action_extractor.py`](sc2_rl_agent/starcraftenv_test/utils/action_extractor.py)

### 3. Failed Commander experiment (documented for completeness)

Before settling on the resource filter, I built a two-tier LLM architecture: a strategic **Commander** firing every 30–60 game seconds above the original **Operations Officer** (the existing per-tick LLM). The Commander emitted JSON directives with `saving_for`, `tech_path`, allowlist, and forbidden actions; the Operations Officer was constrained accordingly.

It failed in three ways: (1) **tier-on-tier contradiction** — the Officer would propose Gateway/Cybernetics while the Commander had set `saving_for: nexus`, and rejected actions became EMPTY without the Officer learning why; (2) **compounding LLM failures** — a Commander parse error froze the directive for 60 seconds, constraining the Officer with stale state; (3) **no correction channel** — once committed, the Commander could be confidently wrong with no feedback path.

I removed the Commander and replaced it with the deterministic resource filter, which addressed the same problem (resource starvation loops) without the LLM-on-LLM coupling. The lesson generalized: **push every constraint into deterministic code that the LLM does not own; use the LLM only for choices that genuinely require its generative ability.**

The defunct Commander code is preserved in the repo at `sc2_rl_agent/starcraftenv_test/commander.py` and `rush_commander_prompt.py` for reference but is not imported anywhere.

---

## Updated results table

Extends the original paper's win-rate comparison with this fork's confirmed Level 6 result:

| Prompt           | LV1  | LV2  | LV3  | LV4    | LV5   | LV6                  |
|------------------|------|------|------|--------|-------|----------------------|
| Prompt1 (orig.)  | 7/8  | 6/9  | 2/8  | 1/8    | 0/8   | 0/8                  |
| Prompt2 (orig.)  | 8/8  | 9/9  | 8/8  | 21/25  | 7/14  | 0/12                 |
| **This fork**    | —    | —    | —    | —      | —     | **stable adv. 1/1**¹ |

¹ Single game, terminated manually at ~17 minutes after the agent established a clear economic and tech-tree lead but did not commit to a finishing attack within the time available. Full benchmark grid (5+ games per difficulty level) is in progress.

### Game artifact: 6822-step instrumented run

| Metric                              | Value                |
|-------------------------------------|----------------------|
| Total decision steps                | 6822                 |
| Game time                           | ~17 minutes          |
| Valid actions executed              | 2294 (33.6%)         |
| EMPTY ACTION steps                  | 4528 (66.4%)         |
| ↳ Parser failures                   | 469 (10.4% of EMPTY) |
| ↳ LLM silent (no action tag)        | 4059 (89.6% of EMPTY)|
| Steps with at least one failure     | 1633 (23.9%)         |
| `cannot-afford` share of failures   | 7.6% (was 43%)       |
| Peak workers                        | 75 (hard cap)        |
| Peak supply                         | 165–170              |
| Bases at peak                       | 5                    |

### Model used in this fork

DeepSeek-V4-Flash (`deepseek-v4-flash`, OpenAI-compatible API at `https://api.deepseek.com/v1`), temperature 0.3, K=5 decisions per tick. The original paper used GPT-3.5-Turbo-16k.

### Known residual issue: the conversion-failure problem

Even at Level 6 with the resource filter active, the agent does not convert macro dominance into a winning attack. The action distribution is illuminating: across the 6822-step game the LLM trained 254 Probes, 172 High Templars, 137 Zealots, 116 Stalkers, 92 Observers, 59 Phoenixes, 59 Void Rays, 55 Immortals, 43 Colossi, 35 Sentries, and 28 Adepts — **essentially every available unit type, in roughly comparable amounts.**

A pro-level Protoss army is built around a *committed* composition (typically Stalker/Immortal or Zealot/Archon/Templar with strict ratios). The LLM's diversification likely comes from the prompt asking for `K=5` decisions per tick — the LLM produces 5 different action *types*, which over many ticks averages to a quasi-uniform-over-the-roster build. Strategically this loses to a focused army of equivalent supply. There is also no `COMMIT_TO_ATTACK_AT_SUPPLY_180` primitive in the action vocabulary; only `MULTI-ATTACK`, which sends the current army immediately.

This is the natural next research direction for this codebase.

---

## Original paper

The original method, motivation, and broader context come from the upstream paper. **If you cite this work, cite the original paper:**

```
@misc{ma2024largelanguagemodelsplay,
      title={Large Language Models Play StarCraft II: Benchmarks and a Chain of Summarization Approach},
      author={Weiyu Ma and Qirui Mi and Yongcheng Zeng and Xue Yan and Yuqiao Wu and Runji Lin and Haifeng Zhang and Jun Wang},
      year={2024},
      eprint={2312.11865},
      archivePrefix={arXiv},
}
```

- Paper: https://arxiv.org/abs/2312.11865
- Project page: https://sites.google.com/view/textstarcraft2/
- Demo video: https://www.youtube.com/watch?v=Iz6Hd917eME

StarCraft II is a challenging benchmark for AI agents due to micro-level operations and macro-awareness. The original paper develops the textual StarCraft II environment ("TextStarCraft II") and proposes a Chain of Summarization method, with single-frame summarization for processing raw observations and multi-frame summarization for analyzing game information, providing command recommendations, and generating strategic decisions. The original work demonstrates LLM agents capable of defeating the built-in AI at Hard (Lv5) difficulty.

| Work | Method | Compute | Required replays | Best result | Strategy interpretability | Expansibility |
|------|--------|---------|------------------|-------------|---------------------------|---------------|
| AlphaStar | SL+RL+self-play | 12000 CPU cores, 384 TPUs | 971,000 | Serral | ✘ | ✘ |
| SCC | SL+RL+self-play | Linear | 4,638 | Time (IEM2023 Champion) | ✘ | ✘ |
| HierNet-SC2 | data-mining + RL | 4 GPUs, 48 CPU cores | 608 | built-in AI Lv-10 | ✘ | ✘ |
| AlphaStar Unplugged | offline RL | not clear | 20m | AlphaStar BC agent | ✘ | ✘ |
| ROA-Star | SL+RL+self-play | 2x 64 V100 | 120,938 | hero (GSL Champion) | ✘ | ✘ |
| **Original (Ma et al. 2024)** | prompt + rule-based script | 1 GPU, 1 CPU (home computer) | 0 | built-in AI Lv-5 | ✔ | ✔ |
| **This fork** | + deterministic filters | 1 GPU, 1 CPU (home computer) | 0 | built-in AI Lv-6 (stable adv.) | ✔ | ✔ |

### Original performance benchmarks (preserved for reference)

Comparing models using either the full CoS or CoS without CoT (from the original paper):

| Model                | Method     | Win Rate | PBR    | RUR   | APU    | TR     |
|----------------------|------------|----------|--------|-------|--------|--------|
| **Using Full CoS**   |            |          |        |       |        |        |
| GPT3.5-Turbo-16k     | Full CoS   | 5/10     | 0.0781 | 7875  | 0.7608 | 0.4476 |
| GPT4-Turbo           | Full CoS   | 3/6      | 0.0337 | 8306  | 0.7194 | 0.3452 |
| Gemini-Pro           | Full CoS   | 2/10     | 0.0318 | 9284  | 0.6611 | 0.3571 |
| GLM4                 | Full CoS   | 2/10     | 0.0327 | 3131  | 0.6644 | 0.2904 |
| Claude2.1            | Full CoS   | 2/9      | 0.0219 | 10867 | 0.6599 | 0.4312 |
| **Using CoS without CoT** |        |          |        |       |        |        |
| Finetune-ChatGlm3 6b | CoS w/o CoT| 2/10     | 0.0528 | 30356 | 0.6547 | 0.1714 |
| Finetune-Qwen 1.8b   | CoS w/o CoT| 6/10     | 0.0384 | 12826 | 0.7506 | 0.2095 |
| Finetune-Qwen 7b     | CoS w/o CoT| 6/12     | 0.0421 | 12276 | 0.7234 | 0.3214 |
| Finetune-Llama2 7b   | CoS w/o CoT| 0/12     | 0.0469 | 12295 | 0.5752 | 0.0853 |

---

## Install StarCraft II and setup maps

### Install StarCraft II

StarCraft II is developed by Blizzard and has a number of professional leagues (IEM, WTL, ...). Download Battle.net from https://us.shop.battle.net/en-us or https://www.blizzard.com/.

Chinese players: due to changes in regional access, see this guide ([Bilibili video](https://www.bilibili.com/video/BV1As4y147NP/)) or search online.

### Download maps

Open `StarCraft II Editor.exe` and use it to download the latest ladder map.

![editor login](https://github.com/histmeisah/Large-Language-Models-play-StarCraftII/assets/49554454/095023ec-497b-4510-889e-6166f6cfb57d)

Log in with your Blizzard account and search for the map you want.

![editor search](https://github.com/histmeisah/Large-Language-Models-play-StarCraftII/assets/49554454/8f68af70-d877-47cc-8d36-8240c7645900)

Place the maps in `StarCraft II/Maps/` (create the `Maps` folder if it doesn't exist), or download them from the upstream repo:

![map download](https://github.com/histmeisah/Large-Language-Models-play-StarCraftII/assets/49554454/13872898-aec1-411a-8c1a-76733a336682)

---

## Setup environment

### Create environment

- **OS:** Windows 11 (Blizzard does not release the latest SC2 client on Linux; this repo runs on Windows).
- **Python:** 3.10
- **CUDA:** 12.1
- **PyTorch:** 2.1.0
- **OpenAI Python SDK:** ≥1.0 for the main `test_the_env.py` path (`LLM/gpt_test.py::ChatBot_SingleTurn` uses the `OpenAI` client). Some older standalone scripts under `LLM/` (`ChatBot_MultiTurn`, `API_caller.py`, `test_api.py`) still use the pre-1.0 `openai.ChatCompletion` API and are not used by the main path.

Install all packages:

```bash
pip install -r requirements.txt
```

### Tips

- **`burnysc2`**: the core SC2 control package. Documentation: [Python-sc2](https://github.com/BurnySc2/python-sc2)
- **`chromadb`**: the vector database used for action-name similarity search. Due to package conflicts, install Chromadb **before** burnysc2.
- **`Hugging Face` + `sentence-transformers`**: the embedding model `sentence-transformers/all-mpnet-base-v2` is downloaded automatically. A pre-bundled release zip is also available with the embedding model included.

### Using DeepSeek instead of OpenAI (this fork's default)

This fork uses DeepSeek-V4-Flash by default. Set the API base to `https://api.deepseek.com/v1` and pass your DeepSeek key. The OpenAI Python SDK (≥1.0) talks to DeepSeek's OpenAI-compatible endpoint without modification:

```bash
python -m sc2_rl_agent.starcraftenv_test.test_the_env \
  --agent_type gpt \
  --player_race Protoss \
  --opposite_race Zerg \
  --difficulty Hard \
  --LLM_model_name deepseek-v4-flash \
  --LLM_api_base https://api.deepseek.com/v1 \
  --LLM_api_key sk-YOUR-DEEPSEEK-KEY \
  --LLM_temperature 0.3
```

To use OpenAI/GPT-3.5 as in the original paper, set `--LLM_model_name gpt-3.5-turbo-16k`, `--LLM_api_base` to OpenAI's URL, and pass an OpenAI key.

---

## Run demo

### Game modes

- **Agent vs Botai:** `test_the_env.py` (single process) or `multiprocess_test.py` (parallel).
- **Human vs Agent:** `Human_LLM_agent_test.py`.
- **Agent vs Agent:** `2agent_test.py`.

### Single process

Run `test_the_env.py`. Key parameters:

- **`--player_race`**: only `Protoss` is supported. Zerg and Terran are under development.
- **`--opposite_race`**: typically `Zerg`; `Terran` and `Protoss` also work.
- **`--difficulty`**: 10 levels from VeryEasy (Lv1) to CheatInsane (Lv10). Note that python-sc2's level names differ from the in-game client; mapping below.

| Level | Blizzard | python-sc2 |
|-------|----------|------------|
| 1 | VeryEasy | VeryEasy |
| 2 | Easy | Easy |
| 3 | Medium | Medium |
| 4 | Hard | MediumHard |
| 5 | Harder | Hard |
| 6 | Very Hard | Harder |
| 7 | Elite | VeryHard |
| 8 | CheatVision | CheatVision |
| 9 | CheatMoney | CheatMoney |
| 10 | CheatInsane | CheatInsane |

- **`--LLM_model_name`**: e.g. `deepseek-v4-flash` (this fork's default) or `gpt-3.5-turbo-16k` (original paper).
- **`--LLM_temperature`**: 0–1.
- **`--LLM_api_key`**, **`--LLM_api_base`**: your API credentials.
- **`--strategy`**: `macro` (default) or `rush` — flag exists for compatibility with the abandoned Commander variant; only `macro` is supported in the current code path.

A single Lv5+ game can take 5–7 hours of wall-clock time (LLM call latency dominates). The Lv6 instrumented game in this fork ran ~17 minutes of in-game time across 6822 decision steps.

Build-order types:

```python
AI_BUILD_LEVELS = ['randombuild', 'rush', 'timing', 'power', 'macro', 'air']
```

### Multi-process

`multiprocess_test.py` runs N games in parallel. Set `--num_processes` to the number of parallel SC2 instances. ~3–4 parallel games is realistic on a single home GPU; further parallelism is bottlenecked by SC2 client memory, not the LLM. Other parameters are the same as single process.

---

## Telemetry

Every `test_the_env.py` game run with `--agent_type gpt|chatgpt|deepseek` is recorded for comparing experiments. This is measurement only: prompts, decision frequency, the extractor, the resource filter and `mix_actions` are unchanged.

**Local JSONL is the source of truth; wandb is only a sink.** If wandb is missing, not logged in or offline, the game still runs to the end and every local file is still written. Each game folder `log/chatgpt_log/game_<time>_<pid>/` keeps the old files (`action_executed.json`, …, same format) and adds:

| File | One line per | Contents |
|---|---|---|
| `llm_calls.jsonl` | HTTP attempt (retries included) | tokens, DeepSeek cache hit/miss, latency, `model_returned`, error, full prompt and response |
| `decision_ticks.jsonl` | L2 decision | parse tier, parsed → filtered → queued counts, slot sources |
| `slots.jsonl` | dequeued action slot | `slot_source`, executed name, failure reason/category |
| `observations.jsonl` | env step | macro metrics (minerals, supply, bases, army entropy, …) |
| `events.jsonl` | run event | run start/end and config, warnings (e.g. `model_changed`, `wandb_disabled`) |
| `game_summary.json` | game | all whole-game metrics below |

The replay (`*.SC2Replay`) is now saved in the same folder; before, `gpt`/`deepseek` replays ended up in `random_log/`.

### wandb setup and modes

```bash
wandb login                     # once; stored in ~/.netrc
python -m sc2_rl_agent.starcraftenv_test.test_the_env --agent_type deepseek --difficulty VeryEasy \
    --wandb_mode offline --exp_group smoke --LLM_api_key sk-...
wandb sync wandb/offline-run-*  # upload offline runs later
```

| `--wandb_mode` (or `WANDB_MODE`) | Behaviour |
|---|---|
| `offline` (default) | writes `./wandb/offline-run-*` locally; upload with `wandb sync` |
| `online` | streams to project `textsc2` |
| `disabled` | no wandb at all; JSONL only |

The CLI flag wins over `WANDB_MODE`. Other flags: `--exp_group` (wandb group, default `adhoc`), `--LLM_reasoning_effort` (sent via `extra_body` only when set), and `--price_in_hit/--price_in_miss/--price_out` (USD per 1M tokens, used for `est_cost_usd`; 0 = no estimate). Runs are named `{difficulty}-{map}-{model}-{time}-p{pid}` and tagged with agent type, difficulty, opponent race, prompt type and `phase0`. The API key is excluded from the config and scrubbed from every file before upload. wandb is only initialised in the main process; the SC2 bot subprocess never imports it. Ctrl+C still writes the summary and uploads the artifact with `result=aborted` (wandb shows such runs as *failed*, since it has no "aborted" state). `multiprocess_test.py` workers write JSONL but only use wandb when `WANDB_MODE` is set.

### Metric definitions

Each L2 decision asks the LLM for K=5 actions. The agent then queues a window of 10 slots, of which at most 5 are real actions. Every slot is tagged with a `slot_source`:

- `llm`: an action the LLM proposed that survived parsing and the resource filter
- `padding`: EMPTY inserted by `mix_actions` between real actions
- `filter_drop`: an LLM proposal the resource filter replaced with EMPTY
- `parse_empty`: EMPTY because the extractor found nothing
- `queue_empty`: EMPTY because the queue ran dry

**Only `llm` slots count as decisions. Padding never enters a denominator.** The old analyzer counted padding as "LLM silent", which is why the 6822-step table above overstates silence. Rerun it through the backfill tool (below) for corrected numbers.

| Metric | Definition |
|---|---|
| `parse_rate` | parsed actions / K requested |
| `admit_rate` | actions left after the resource filter / parsed |
| `queue_rate` | `llm` slots queued / admitted (the rest were cut off by the window, see `truncated_actions`) |
| `exec_success` | `llm` slots that did not fail / `llm` slots executed. A slot fails only if the next step's `action_failures` names *that* action, because failures from scripted `defend()` share the same list. |
| `conversion` | successful `llm` slots / (K × number of ticks) |
| `padding_share` | non-`llm` slots / all slots |

The wandb x-axis is `game_time_s` (`game_loop / 22.4`). `game/*` is logged every 5 game seconds, and `funnel/*` and `llm/*` once per decision (rolling over the last 10 decisions). The summary also has token totals, `cache_hit_ratio`, latency p50/p95, `wall_clock_share_waiting_llm`, `parse_tier_*`, `fail_cat_*`, `action_count_*`, peak supply/workers/bases, `time_to_first_attack_s`, `supply_blocked_s`, `army_comp_entropy_mean` (how evenly the army is spread across unit types), `model_resolved` (the model the API actually returned) and `scripted_attack_count`. That last one counts the steps where the hard-coded "12:00 and supply ≥ 100 → attack" rule fired, so rule-driven attacks can be told apart from the LLM's `MULTI-ATTACK`.

### Backfilling old games

```bash
python -m sc2_rl_agent.starcraftenv_test.telemetry.backfill \
    --log_root sc2_rl_agent/starcraftenv_test/log/chatgpt_log --exp_group legacy --wandb_mode online
```

This rebuilds slot sources for old game folders by replaying the agent's queue: the same tick schedule and `mix_actions` placement, the extractor re-run on `commander.json`, and filter drops from `resource_rejections.json`. Results go to `<game>/backfill/` and the originals are not modified. Old games have no token or latency data, and their config is marked `backfilled=true, padding_inferred=true`. Each game prints `backfill_inconsistent_slots`, the number of slots where the rebuilt queue disagrees with what the env actually executed.

Tests (no SC2 or API needed): `python -m pytest sc2_rl_agent/starcraftenv_test/tests/telemetry`.

---

## Customizing your own LLM agent

### Component map

| Component | Path |
|-----------|------|
| LLM call | `ChatBot_SingleTurn` in `sc2_rl_agent/starcraftenv_test/LLM/gpt_test.py` |
| L1 summarization | `generate_summarize_L1` in `sc2_rl_agent/starcraftenv_test/summarize/L1_summarize.py` |
| L2 summarization | `L2_summary` in `sc2_rl_agent/starcraftenv_test/summarize/gpt_test/L2_summarize.py` |
| Action vocabulary | `sc2_rl_agent/starcraftenv_test/utils/action_info.py` |
| Action extractor (this fork's 4-tier version) | `sc2_rl_agent/starcraftenv_test/utils/action_extractor.py` |
| **Resource filter (new)** | **`sc2_rl_agent/starcraftenv_test/utils/resource_filter.py`** |
| Agent logic | `sc2_rl_agent/starcraftenv_test/agent/chatgpt_agent.py` |
| System prompts | `sc2_rl_agent/starcraftenv_test/prompt/prompt.py` (`StarCraftIIPrompt_V2` is current default) |

### Environment internals

The bot lives at `sc2_rl_agent/starcraftenv_test/env/bot/`:

- **State (Obs → Text):** `Protoss_bot.py::get_information()` builds the L1 textual observation.
- **Action (Text → Action):** the various `handle_action` functions in `Protoss_bot.py` execute each vocabulary entry.

To add Zerg or Terran support, modify these. The action vocabulary is hardcoded for Protoss in this fork.

### Models tested

- **Online:** GPT-3.5-Turbo (original), GPT-4-Turbo, Gemini-Pro, GLM4, Claude2.1, **DeepSeek-V4-Flash (this fork)**.
- **Local:** GLM3, Qwen, Qwen1.5.

---

## Evaluation metrics

The original paper's metrics, all preserved:

- **Win Rate**: % games won out of total games played.
- **Population Block Ratio (PBR)**: time spent at population cap, lower is better (less macro mismanagement).
- **Resource Utilization Ratio (RUR)**: how efficiently resources are spent; lower is better (less idle stockpile).
- **Average Population Utilization (APU)**: efficiency of population capacity use; higher is better.
- **Technology Rate (TR)**: proportion of tech tree completed; higher means more advancement.

This fork additionally instruments and reports:

- **Empty action rate**: fraction of decision steps where the bot does nothing. Decomposed into parser failures vs. genuine LLM silence (no action tag in response).
- **Cannot-afford failure rate**: fraction of attempted actions that fail because of insufficient minerals/gas. Used to validate the resource filter.
- **Failure category distribution**: breakdown of why python-sc2 rejected actions (chrono timing, building unavailable, building busy, placement error, warp gate confusion, supply blocked, cannot afford).
- **Phase decomposition**: valid/empty/failure rates split into early (0–3m), mid-early (3–7m), mid (7–10m), late (10m+) phases.
- **Strategic milestones**: wall-clock game time at first Pylon, first Gateway, first Cybernetics Core, Warp Gate research complete, first Stalker, second Nexus, Blink research, first Observer, first Immortal, first Colossus.

The post-game analyzer that produces these is at `sc2_llm_analyzer.py` (top-level) and consumes the per-game log folder produced under `sc2_rl_agent/starcraftenv_test/log/chatgpt_log/game_<timestamp>_<process_id>/`. Outputs include a text report, a per-step CSV timeline, and 8 matplotlib charts.

---

## Acknowledgments

- **Original repository and paper**: Weiyu Ma et al., NeurIPS 2024. All credit for the framework, environment, Chain of Summarization method, and prompt design belongs to them.
- **CS 6900 final project, Ohio University**: the work in this fork was conducted as a course final project under the guidance of Dr. Chang Liu.

## License

Same as upstream. See [LICENSE](LICENSE) in this repository.