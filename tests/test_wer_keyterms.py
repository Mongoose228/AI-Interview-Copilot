"""Unit tests for WER helpers and keyterm extraction."""

from interview_copilot.eval.wer import cer, normalize_text, term_hits, wer
from interview_copilot.stt.keyterms import build_initial_prompt, extract_keyterms


def test_wer_perfect():
    assert wer("hello world", "hello world") == 0.0


def test_wer_substitution():
    assert wer("hello world", "hello there") == 0.5


def test_wer_empty_ref_and_hyp():
    assert wer("", "") == 0.0


def test_cer_basic():
    assert cer("abc", "abc") == 0.0
    assert cer("abc", "abd") > 0.0


def test_normalize_text():
    assert normalize_text("Hello, World!") == "hello world"


def test_term_hits():
    hits = term_hits(
        "Explain Kubernetes and Redis",
        "Explain Kubernetes and red is",
        ["Kubernetes", "Redis"],
    )
    assert hits["Kubernetes"] is True
    assert hits["Redis"] is False


def test_extract_keyterms():
    text = "# Profile\nSkills: Kubernetes, PostgreSQL, Redis\nWorked on idempotency."
    terms = extract_keyterms(text)
    joined = " ".join(terms).lower()
    assert "kubernetes" in joined
    assert "postgresql" in joined


def test_build_initial_prompt():
    prompt = build_initial_prompt("Skills: Kubernetes PostgreSQL")
    assert "Kubernetes" in prompt or "kubernetes" in prompt.lower()
