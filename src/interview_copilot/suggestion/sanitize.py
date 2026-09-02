"""Basic prompt injection mitigation.

Filters known injection patterns from user-provided text before
inserting it into LLM prompts. This is a defence-in-depth measure,
not a complete solution.
"""

import re

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"assistant\s*:\s*",
    r"<\|.*?\|>",            # special tokens like <|im_start|>
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def sanitize_for_prompt(text: str) -> str:
    """Remove known injection patterns from text.

    Replaces matched patterns with [FILTERED] to preserve text length
    and readability while neutralising the injection.
    """
    result = text
    for pattern in _COMPILED:
        result = pattern.sub("[FILTERED]", result)
    return result
