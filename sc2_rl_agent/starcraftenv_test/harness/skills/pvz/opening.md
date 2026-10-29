---
name: pvz-opening
phase: early (0:00-4:00)
description: Safe gateway expand into warp gate; scout to choose the mid-game skill.
use_when: Game start against Zerg, before any enemy army has been seen.
---
# PvZ opening: gateway expand

## Goal
Reach two bases with constant probe production, warp gate research and a small
defensive force, and scout what Zerg is doing before 3:00.

## Order (action ids)
1. Probes nonstop (0) until 22 per base; chrono the Nexus while it trains (66).
2. Pylon (19) at ~14 supply, Gateway (22), Assimilator (20).
3. Scout with one probe (60) right after the first Gateway; it reveals
   pool timing, gas and early lings.
4. Second Nexus (21) around 1:30-2:00 if no early lings were seen.
5. Cybernetics Core (23), second Assimilator (20), Warp Gate research (34).
6. 1-2 Stalkers (3) or Adepts (2) for scouting and defense, one Shield Battery (33)
   at the natural if lings were seen.

## Army goal
mode defend, units {"stalker": 2, "adept": 1}. Production stays mostly on probes.

## Switch skill when
- 6+ zerglings or any roach/baneling before 3:30 -> `pvz-hold-early-rush`
- roaches / roach warren seen -> `pvz-vs-roach-ravager`
- many zerglings or banelings / baneling nest -> `pvz-vs-ling-bane`
- hydralisks, lurkers, hydra/lurker den -> `pvz-vs-hydra-lurker`
- mutalisks or a spire -> `pvz-vs-mutalisk`
- otherwise at ~4:00 pick the counter to what you have seen, or `pvz-late-max`

## Bot notes
The bot cannot micro or cast spells. Sentries' force field / guardian shield and
Oracles' beam are never used, so do not open with them.
