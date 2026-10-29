---
name: pvz-late-max
phase: late (10:00+)
description: Generic late game: 3-4 bases, 3/3 upgrades, chargelot/immortal/archon or skytoss at 190+ supply.
use_when: Late game with a mixed or unknown Zerg army, or ultralisks/hive tech appear.
---
# PvZ late game: maxed army

## Goal
Bank on a 4-base economy, finish upgrades, attack with a full army and
re-max quickly from many gateways.

## Tech path (action ids)
1. Fourth Nexus (21) when the third is saturated; probes up to ~70 (0).
2. Two Forges (24): ground weapons/armor/shields to level 3 (44-52), which
   need the Twilight Council (25).
3. Charge (43); Templar Archives (28) for archons (5 -> 18).
4. Two Robotics Facilities (26) making Immortals (14) against ultralisks and roaches.
5. 10-12 gateways (22) for re-max warp-ins.

## Army goal
mode attack, units {"zealot": 24, "immortal": 6, "archon": 6, "observer": 2}, attack_at_supply 190.
Against heavy air instead, follow `pvz-vs-air-late`.

## When to attack / defend
- MULTI-ATTACK (64) at 190+ supply with at least +2 weapons.
- After a trade, keep the goal and warp in again; attack again at 180+.

## Bot notes
The bot spreads the army to enemy expansions once the enemy main is cleared;
keep producing so reinforcements keep arriving.
