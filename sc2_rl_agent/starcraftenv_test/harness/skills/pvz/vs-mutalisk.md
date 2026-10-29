---
name: pvz-vs-mutalisk
phase: mid (5:00-12:00)
description: Phoenix/stalker/archon with static defense against mutalisk harassment.
use_when: Mutalisks or a spire were seen.
---
# PvZ vs mutalisk

## Why
Mutalisks harass mineral lines and avoid fights. Phoenixes are faster and
out-range them; archons' splash and stalkers' anti-air protect the main army;
cannons and batteries protect probes while the army is away.

## Tech path (action ids)
1. Stargate (27), a second one (27) when gas allows; Phoenix (11) from both
   (the bot caps phoenixes at 4 at a time in production).
2. Fleet Beacon (31) -> Phoenix range (58) once 6+ phoenixes are out.
3. Forge (24) -> one Photon Cannon (32) and one Shield Battery (33) per mineral line.
4. Templar Archives (28) -> High Templar (5) -> Archon (18); stalkers (3) from gateways.
5. Air weapons (35) if the game goes long.

## Army goal
mode defend, units {"phoenix": 6, "stalker": 10, "archon": 2}, attack_at_supply 120.
Switch the army goal to mode attack once phoenixes outnumber mutalisks seen.

## When to attack / defend
- Stay home until phoenixes >= mutalisks seen; then MULTI-ATTACK (64).
- If the Zerg moves to corruptors/brood lords, switch to `pvz-vs-air-late`.

## Bot notes
Phoenix graviton beam is never used; phoenixes are anti-air only here.
