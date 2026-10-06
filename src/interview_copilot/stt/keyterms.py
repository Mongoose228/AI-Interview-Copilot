"""Extract keyterms / initial prompts from candidate profiles."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+.#/-]{1,40}")
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "have",
    "been",
    "will",
    "your",
    "about",
    "into",
    "over",
    "under",
    "skills",
    "experience",
    "education",
    "profile",
    "summary",
    "achievements",
}


def extract_keyterms(profile_text: str, *, limit: int = 50) -> list[str]:
    """Pull likely tech tokens / proper nouns from markdown profile text."""
    if not profile_text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _WORD.finditer(profile_text):
        token = match.group(0)
        lower = token.lower()
        if lower in _STOP:
            continue
        # Prefer CamelCase, ALLCAPS, or tokens with digits / punctuation.
        interesting = (
            any(c.isupper() for c in token[1:])
            or token.isupper()
            or any(c.isdigit() for c in token)
            or any(c in "+.#/-" for c in token)
            or len(token) >= 6
        )
        if not interesting:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered


def build_initial_prompt(profile_text: str, *, limit: int = 30) -> str:
    terms = extract_keyterms(profile_text, limit=limit)
    if not terms:
        return (
            "Technical software engineering interview. "
            "Prefer correct spelling of APIs, databases, and frameworks."
        )
    joined = ", ".join(terms)
    return (
        "Technical software engineering interview vocabulary: "
        f"{joined}."
    )
