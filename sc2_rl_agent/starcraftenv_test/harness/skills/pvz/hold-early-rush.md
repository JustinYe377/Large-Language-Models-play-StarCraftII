---
name: pvz-hold-early-rush
phase: early (0:00-5:00)
description: Hold zergling/roach/baneling rushes with batteries, zealots and stalkers, then counter.
use_when: 6+ zerglings or any roach/baneling before 3:30, or enemy units near our bases early.
---
# PvZ: hold an early rush

## Goal
Lose nothing important to the first attack, then take the game back with a
stronger economy. The built-in AI attacks in waves; surviving one wave buys time.

## Order (action ids)
1. Keep probes going (0) but do not take a third base yet.
2. Shield Batteries (33) next to the natural Nexus: 2 against roaches, 1-2 against lings.
   They need a Cybernetics Core; a Photon Cannon (32) needs a Forge (24).
3. Gateway units from every gateway: Zealots (1) against lings, Stalkers (3) or
   Adepts (2) against roaches. Add a second Gateway (22) if minerals pile up.
4. Warp Gate research (34) as soon as the Cybernetics Core finishes; chrono it (67).
5. Keep the army at home (MULTI-RETREAT 65 if it has walked out); the bot's
   defend routine engages enemies that come close to a Nexus.

## Army goal
mode defend, units {"zealot": 4, "stalker": 4, "adept": 2}.

## Then
When the wave is dead and there are no enemy units near our bases, expand (21)
and switch to the counter skill for what you saw: roaches ->
`pvz-vs-roach-ravager`, lings/banes -> `pvz-vs-ling-bane`.

## Bot notes
Do not MULTI-ATTACK with fewer than ~8 units; the bot sends idle units straight
at the enemy main and they arrive one by one.
