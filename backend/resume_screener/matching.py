"""Match score: which of a job's required skills a candidate has. Pure Python, no model."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Different spellings of the same skill. Keys and values are already normalised.
ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "postgres": "postgresql",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "powerbi": "power bi",
    "sklearn": "scikit-learn",
    "k8s": "kubernetes",
    "gcp": "google cloud",
    "amazon web services": "aws",
}


def normalize_skill(skill: str) -> str:
    s = re.sub(r"\s+", " ", skill.strip().lower())
    s = s.rstrip(".,;:")  # trailing punctuation only, so "c++" and "c#" survive
    return ALIASES.get(s, s)


@dataclass
class SkillMatch:
    score: int  # 0-100
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def match_skills(required: list[str], candidate_skills: list[str]) -> SkillMatch:
    have = {normalize_skill(s) for s in candidate_skills}
    seen: set[str] = set()
    matched: list[str] = []
    missing: list[str] = []
    for skill in required:
        key = normalize_skill(skill)
        if not key or key in seen:
            continue
        seen.add(key)
        (matched if key in have else missing).append(skill)
    total = len(matched) + len(missing)
    score = round(100 * len(matched) / total) if total else 0
    return SkillMatch(score, matched, missing)


def parse_years(phrase: str) -> float | None:
    """'5+ years' -> 5, '2.5 yrs' -> 2.5. Words like 'eighteen' are not parsed."""
    m = re.search(r"\d+(?:\.\d+)?", phrase)
    if not m:
        return None
    value = float(m.group())
    return int(value) if value.is_integer() else value
