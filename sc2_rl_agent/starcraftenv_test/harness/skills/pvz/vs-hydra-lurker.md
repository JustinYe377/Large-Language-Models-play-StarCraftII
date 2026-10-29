---
name: pvz-vs-hydra-lurker
phase: mid-late (6:00-14:00)
description: Colossus/immortal/chargelot army with observers against hydralisk/lurker.
use_when: Hydralisks are the main Zerg army, or lurkers / a lurker den / hydralisk den were seen.
---
# PvZ vs hydralisk / lurker

## Why
Colossi out-range hydralisks and kill them in groups; immortals handle lurkers
and roaches mixed in; zealots tank. Lurkers are invisible while burrowed, so
the army must bring observers or it cannot even hit them.

## Tech path (action ids)
1. Two Robotics Facilities (26), Robotics Bay (30), Colossus (16) nonstop,
   Extended Thermal Lance (53) as soon as the bay finishes.
2. 2 Observers (13) at all times when lurkers were seen.
3. Forge (24): +1 / +2 ground weapons (44, 45) and armor (47, 48).
4. Twilight Council (25) -> Charge (43) for the zealot screen.
5. Third base (21) by ~6:30 behind the colossus count.

## Army goal
mode attack, units {"colossus": 4, "immortal": 3, "zealot": 12, "observer": 2}, attack_at_supply 150.

## When to attack / defend
- MULTI-ATTACK (64) at the goal with Extended Thermal Lance done.
- Corruptors appearing in numbers counter colossi: add stalkers (3) or switch to
  `pvz-vs-air-late`.

## Bot notes
Disruptors are never fired by the bot; do not build them.
