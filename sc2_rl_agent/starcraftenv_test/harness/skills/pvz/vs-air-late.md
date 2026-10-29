---
name: pvz-vs-air-late
phase: late (10:00+)
description: Tempest/void ray/carrier sky army with stalkers against corruptor/brood lord/viper.
use_when: Brood lords, many corruptors (6+), vipers or a greater spire were seen.
---
# PvZ vs corruptor / brood lord

## Why
Brood lords out-range ground armies and corruptors hard-counter colossi.
Tempests out-range brood lords, void rays shred corruptors (armored), carriers
overwhelm both, and stalkers kill the slow brood lords on the ground.

## Tech path (action ids)
1. 2-3 Stargates (27), Fleet Beacon (31).
2. Void Rays (7) first, then Tempests (9) against brood lords and Carriers (8)
   once the economy is on 3+ bases.
3. Air weapons and armor (35-40) up to level 2-3; Void Ray speed (57).
4. Keep stalkers (3) from the gateways; stop colossus production.

## Army goal
mode attack, units {"voidray": 6, "tempest": 4, "carrier": 3, "stalker": 10}, attack_at_supply 170.

## When to attack / defend
- Defend while the sky army is under 50% of the goal; the slow brood lords
  are weakest when they come to us.
- MULTI-ATTACK (64) at the goal with +2 air weapons.

## Bot notes
Mass recall and void ray prismatic alignment are never used; the army moves
as a group only because the bot orders all idle units together.
