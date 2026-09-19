"""Tests for the debounce/accumulator logic via orchestrator."""
import asyncio
import time
import uuid

import pytest

from interview_copilot.models import Transcript
from interview_copilot.pipeline import InterviewPipeline
from tests.fakes import (
    FakeAudioCapture,
    FakeProfileManager,
    FakeSuggester,
    FakeVAD,
    FakeWhisperEngine,
)


def _make_transcript(text: str) -> Transcript:
    return Transcript(
        phrase_id=uuid.uuid4(),
        text_en=text,
        language="en",
        confidence=0.95,
        stt_duration_s=0.5,
        stt_started_at=time.time(),
        stt_ended_at=time.time() + 0.5,
    )


@pytest.mark.asyncio
async def test_multiple_segments_merge_into_one_card(monkeypatch, sample_profile):
    monkeypatch.setattr(
        "interview_copilot.pipeline.config.LLM_DEBOUNCE_DELAY_S", 0.05
    )

    suggester = FakeSuggester(configured=True, answer="Merged answer")
    pipeline = InterviewPipeline(
        audio=FakeAudioCapture(),
        vad=FakeVAD(),
        stt=FakeWhisperEngine(),
        translator=None,
        suggester=suggester,
        profile_mgr=FakeProfileManager(sample_profile),
    )

    cards = []
    suggestions = []

    pipeline.set_callbacks(
        transcript_callback=lambda r: cards.append(r),
        suggestion_callback=lambda r: suggestions.append(r),
        token_callback=lambda *_: None,
        error_callback=lambda *a, **k: None,
        audio_status_callback=lambda *a: None,
    )

    pipeline._stop_event.clear()
    pipeline._loop = asyncio.get_running_loop()
    pipeline.transcript_queue = asyncio.Queue()

    orch = asyncio.create_task(pipeline._async_orchestrator())

    await pipeline.transcript_queue.put(_make_transcript("What is"))
    await pipeline.transcript_queue.put(
        _make_transcript("the time complexity of quicksort?")
    )

    # Wait for debounce + processing
    for _ in range(50):
        if suggestions:
            break
        await asyncio.sleep(0.05)

    pipeline._stop_event.set()
    await asyncio.sleep(0.1)
    orch.cancel()
    try:
        await orch
    except asyncio.CancelledError:
        pass

    assert len(cards) == 1
    assert "What is the time complexity of quicksort?" in cards[0].transcript
    assert len(suggester.calls) == 1
    assert suggestions


@pytest.mark.asyncio
async def test_filler_not_sent_to_llm(monkeypatch, sample_profile):
    monkeypatch.setattr(
        "interview_copilot.pipeline.config.LLM_DEBOUNCE_DELAY_S", 0.05
    )

    suggester = FakeSuggester(configured=True)
    pipeline = InterviewPipeline(
        audio=FakeAudioCapture(),
        vad=FakeVAD(),
        stt=FakeWhisperEngine(),
        translator=None,
        suggester=suggester,
        profile_mgr=FakeProfileManager(sample_profile),
    )

    pipeline.set_callbacks(
        transcript_callback=lambda r: None,
        suggestion_callback=lambda r: None,
        token_callback=lambda *_: None,
        error_callback=lambda *a, **k: None,
        audio_status_callback=lambda *a: None,
    )

    pipeline._stop_event.clear()
    pipeline._loop = asyncio.get_running_loop()
    pipeline.transcript_queue = asyncio.Queue()

    orch = asyncio.create_task(pipeline._async_orchestrator())
    await pipeline.transcript_queue.put(_make_transcript("okay"))
    await asyncio.sleep(0.2)

    pipeline._stop_event.set()
    orch.cancel()
    try:
        await orch
    except asyncio.CancelledError:
        pass

    assert suggester.calls == []


def test_transcript_merge_preserves_last_id():
    t1 = _make_transcript("What is")
    t2 = _make_transcript("the answer?")
    merged = Transcript(
        phrase_id=t2.phrase_id,
        text_en=f"{t1.text_en} {t2.text_en}",
        language="en",
        confidence=min(t1.confidence, t2.confidence),
        stt_duration_s=t1.stt_duration_s + t2.stt_duration_s,
    )
    assert merged.phrase_id == t2.phrase_id
    assert merged.text_en == "What is the answer?"
