---
name: pvz-vs-ling-bane
phase: mid (4:00-10:00)
description: Chargelot/archon/colossus army against zergling/baneling floods.
use_when: Mass zerglings (16+) or banelings / a baneling nest are the main Zerg army.
---
# PvZ vs zergling / baneling

## Why
Lings and banes are cheap, fast and splash-vulnerable. Zealots with Charge trade
evenly and absorb banelings cheaply; archons and colossi kill lings in bulk.
Stalkers and adepts melt to banelings, so keep them few.

## Tech path (action ids)
1. Twilight Council (25) -> Charge (43) first; Forge (24) -> +1 armor (47), then +1 shields (50).
2. Robotics Facility (26) -> Robotics Bay (30) -> Colossus (16) -> Extended
   Thermal Lance (53).
3. Templar Archives (28) -> High Templar (5) -> Archon (18) once gas allows.
4. 6-8 gateways (22) by 7:00 for constant zealot warp-ins (1).
5. Photon Cannon (32) or Shield Battery (33) at each mineral line against runbys.

## Army goal
mode attack, units {"zealot": 20, "colossus": 3, "archon": 2, "observer": 1}, attack_at_supply 130.

## When to attack / defend
- Before Charge finishes: defend only.
- MULTI-ATTACK (64) once Charge and 2+ colossi are done; keep warping zealots
  during the attack.
- If ultralisks appear, add immortals (14) and switch to `pvz-late-max`.

## Bot notes
High templar storms are never cast; templar are only worth making as archons
(two templar morph into one archon).
