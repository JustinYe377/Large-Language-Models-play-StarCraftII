---
name: pvz-vs-roach-ravager
phase: mid (4:00-10:00)
description: Immortal/stalker/archon army that beats roach/ravager, attacking at 6 immortals with +1.
use_when: Roaches or ravagers are the main Zerg army, or a roach warren was scouted.
---
# PvZ vs roach / ravager

## Why
Immortals deal bonus damage to armored units (roaches, ravagers) and their
barrier absorbs burst. Stalkers add range and anti-air; archons add splash.
Roach armies have little splash, so a dense immortal army trades very well.

## Tech path (action ids)
1. Robotics Facility (26), then a second one (26) when on two saturated bases.
2. Forge (24) -> +1 ground weapons (44); then +1 armor (47).
3. Observer (13) to scout and to detect burrowed roaches.
4. Twilight Council (25) -> Blink (42) only if gas is floating; Templar Archives
   (28) -> High Templar (5) -> Archon (18) for splash once the third base is up.
5. Third Nexus (21) around 5:30-6:30 once 4+ immortals are out.

## Army goal
mode attack, units {"immortal": 6, "stalker": 12, "archon": 2, "observer": 1}, attack_at_supply 120.
Keep both robotics facilities making immortals (14) and chrono them via the Nexus
when idle gateways are not needed; warp stalkers (3) with spare minerals/gas.

## When to attack / defend
- Defend (stay home, batteries 33) until 4 immortals.
- MULTI-ATTACK (64) when the goal is met and +1 weapons is done. Retreat (65)
  if the army falls below ~60% of the goal supply, rebuild, attack again.
- If hydralisks join in numbers, switch to `pvz-vs-hydra-lurker`.

## Bot notes
Stalker blink is never used, so stalkers are a support unit here, not the core.
