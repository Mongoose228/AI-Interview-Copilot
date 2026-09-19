"""Tests for OpenRouter suggester prompt building and hedging."""

import time
import uuid

import pytest

from interview_copilot.models import ProfileSnapshot, Transcript
from interview_copilot.suggestion.openrouter import (
    OpenRouterSuggester,
    _detect_needs_verification,
)


def test_hedging_detects_uncertainty():
    assert _detect_needs_verification("I'm not sure about that.")
    assert _detect_needs_verification("Maybe Redis would help.")


def test_hedging_ignores_verify_word():
    assert not _detect_needs_verification("I would verify the input data carefully.")


def test_build_system_prompt_contains_rules(monkeypatch):
    monkeypatch.setattr(
        "interview_copilot.suggestion.openrouter.config.OPENROUTER_API_KEY",
        None,
    )
    s = OpenRouterSuggester()
    profile = ProfileSnapshot(
        name="p",
        content_hash="h",
        content="Python expert at Acme",
        loaded_at=time.time(),
    )
    prompt = s._build_system_prompt(profile)
    assert "NEVER invent" in prompt
    assert "Python expert at Acme" in prompt


@pytest.mark.asyncio
async def test_get_suggestion_returns_none_without_client(monkeypatch):
    monkeypatch.setattr(
        "interview_copilot.suggestion.openrouter.config.OPENROUTER_API_KEY",
        None,
    )
    s = OpenRouterSuggester()
    assert not s.is_configured
    result = await s.get_suggestion(
        [Transcript(uuid.uuid4(), "Q?", "en", 1.0, 0.1)],
        ProfileSnapshot("p", "h", "c", time.time()),
    )
    assert result is None
