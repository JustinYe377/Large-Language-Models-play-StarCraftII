"""
Strategy skills: short markdown playbooks the model loads on demand.

Layout: skills/<matchup>/<file>.md with a frontmatter block

    ---
    name: pvz-vs-roach-ravager
    phase: mid (4:00-10:00)
    description: one line
    use_when: scouting signals that make this the right skill
    ---
    body ...

Only the index (name, phase, description, use_when) is always in the prompt;
the body is loaded by the model with the `load_skill` tool (progressive
disclosure), then kept in the prompt as the active skill.

Sources for the strategy content: Liquipedia LotV PvZ guides and community
build guides (see README "Strategy skills"); unit counts are starting points,
adapted to what this bot can execute (it casts no spells).
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

SKILL_ROOT = os.path.dirname(os.path.abspath(__file__))
MATCHUP = {"Zerg": "pvz", "Terran": "pvt", "Protoss": "pvp"}


@dataclass
class Skill:
    name: str
    phase: str
    description: str
    use_when: str
    body: str
    path: str


def _parse(path) -> Optional[Skill]:
    text = open(path, encoding="utf-8").read()
    if not text.startswith("---"):
        return None
    _, head, body = text.split("---", 2)
    meta = {}
    for line in head.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    if "name" not in meta:
        return None
    return Skill(meta["name"], meta.get("phase", ""), meta.get("description", ""),
                 meta.get("use_when", ""), body.strip(), path)


class SkillLibrary:
    def __init__(self, skills: Dict[str, Skill]):
        self.skills = skills

    @classmethod
    def for_opponent(cls, opposite_race) -> "SkillLibrary":
        folder = os.path.join(SKILL_ROOT, MATCHUP.get(str(opposite_race), "none"))
        skills = {}
        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".md"):
                    skill = _parse(os.path.join(folder, fname))
                    if skill:
                        skills[skill.name] = skill
        return cls(skills)

    @property
    def names(self) -> List[str]:
        return list(self.skills)

    def __bool__(self):
        return bool(self.skills)

    def get(self, name) -> Optional[Skill]:
        return self.skills.get(name)

    def index_text(self) -> str:
        return "\n".join(f"- {s.name} [{s.phase}]: {s.description} Use when: {s.use_when}"
                         for s in self.skills.values())
