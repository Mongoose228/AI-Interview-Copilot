"""Word / character error rate helpers for the STT harness."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate. Empty reference + empty hypothesis => 0.0."""
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0 if hyp else 0.0
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference).replace(" ", ""))
    hyp = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0 if hyp else 0.0
    return _edit_distance(ref, hyp) / len(ref)


def term_hits(reference: str, hypothesis: str, terms: list[str]) -> dict[str, bool]:
    hyp_norm = normalize_text(hypothesis)
    out: dict[str, bool] = {}
    for term in terms:
        t = normalize_text(term)
        if not t:
            continue
        # Only score terms that appear in the reference.
        if t not in normalize_text(reference):
            continue
        out[term] = t in hyp_norm
    return out
