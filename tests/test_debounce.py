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


def _make_transcript(
    text: str,
    *,
    phrase_id=None,
    is_final: bool = True,
    revision: int = 0,
) -> Transcript:
    now = time.time()
    return Transcript(
        phrase_id=phrase_id or uuid.uuid4(),
        text_en=text,
        language="en",
        confidence=0.95,
        stt_duration_s=0.5,
        stt_started_at=now,
        stt_ended_at=now + 0.5,
        is_final=is_final,
        revision=revision,
        speech_started_at=now - 1.0,
        speech_ended_at=now,
        audio_duration_s=1.0,
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

    assert len({card.id for card in cards}) == 1
    assert "What is the time complexity of quicksort?" in cards[-1].transcript
    assert len(suggester.calls) == 1
    assert suggestions


@pytest.mark.asyncio
async def test_partial_upserts_immediately_but_only_final_reaches_llm(
    monkeypatch, sample_profile
):
    monkeypatch.setattr(
        "interview_copilot.pipeline.config.LLM_DEBOUNCE_DELAY_S", 0.05
    )
    suggester = FakeSuggester(configured=True, answer="Final answer")
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
        transcript_callback=cards.append,
        suggestion_callback=suggestions.append,
        token_callback=lambda *_: None,
        error_callback=lambda *a, **k: None,
        audio_status_callback=lambda *a: None,
    )
    pipeline._stop_event.clear()
    pipeline._loop = asyncio.get_running_loop()
    pipeline.transcript_queue = asyncio.Queue()
    orch = asyncio.create_task(pipeline._async_orchestrator())

    phrase_id = uuid.uuid4()
    await pipeline.transcript_queue.put(
        _make_transcript(
            "What is asyn",
            phrase_id=phrase_id,
            is_final=False,
            revision=1,
        )
    )
    for _ in range(20):
        if cards:
            break
        await asyncio.sleep(0.01)
    assert len(cards) == 1
    assert cards[-1].is_provisional
    assert suggester.calls == []

    await pipeline.transcript_queue.put(
        _make_transcript(
            "What is asyncio?",
            phrase_id=phrase_id,
            is_final=False,
            revision=2,
        )
    )
    await asyncio.sleep(0.02)
    assert len({card.id for card in cards}) == 1
    assert cards[-1].is_provisional
    assert "asyncio" in cards[-1].transcript
    assert suggester.calls == []

    await pipeline.transcript_queue.put(
        _make_transcript(
            "What is asyncio?",
            phrase_id=phrase_id,
            is_final=True,
            revision=3,
        )
    )
    for _ in range(50):
        if suggestions:
            break
        await asyncio.sleep(0.02)

    pipeline._stop_event.set()
    orch.cancel()
    try:
        await orch
    except asyncio.CancelledError:
        pass

    assert len({card.id for card in cards}) == 1
    assert not cards[-1].is_provisional
    assert len(suggester.calls) == 1
    assert len(pipeline.transcript_history) == 1
    assert pipeline.transcript_history[0].is_final


@pytest.mark.asyncio
async def test_filler_removes_provisional_card(monkeypatch, sample_profile):
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
    cards = []
    removed = []
    pipeline.set_callbacks(
        transcript_callback=cards.append,
        suggestion_callback=lambda *_: None,
        token_callback=lambda *_: None,
        error_callback=lambda *a, **k: None,
        audio_status_callback=lambda *a: None,
        transcript_remove_callback=removed.append,
    )
    pipeline._stop_event.clear()
    pipeline._loop = asyncio.get_running_loop()
    pipeline.transcript_queue = asyncio.Queue()
    orch = asyncio.create_task(pipeline._async_orchestrator())

    phrase_id = uuid.uuid4()
    await pipeline.transcript_queue.put(
        _make_transcript("oka", phrase_id=phrase_id, is_final=False, revision=1)
    )
    await pipeline.transcript_queue.put(
        _make_transcript("okay", phrase_id=phrase_id, is_final=True, revision=2)
    )
    for _ in range(30):
        if removed:
            break
        await asyncio.sleep(0.01)

    pipeline._stop_event.set()
    orch.cancel()
    try:
        await orch
    except asyncio.CancelledError:
        pass

    assert len(cards) == 1
    assert removed == [cards[0].id]
    assert suggester.calls == []


@pytest.mark.asyncio
async def test_empty_filtered_final_removes_provisional_card(
    monkeypatch, sample_profile
):
    monkeypatch.setattr(
        "interview_copilot.pipeline.config.LLM_DEBOUNCE_DELAY_S", 0.05
    )
    pipeline = InterviewPipeline(
        audio=FakeAudioCapture(),
        vad=FakeVAD(),
        stt=FakeWhisperEngine(),
        translator=None,
        suggester=FakeSuggester(configured=True),
        profile_mgr=FakeProfileManager(sample_profile),
    )
    cards = []
    removed = []
    pipeline.set_callbacks(
        transcript_callback=cards.append,
        suggestion_callback=lambda *_: None,
        token_callback=lambda *_: None,
        error_callback=lambda *a, **k: None,
        audio_status_callback=lambda *a: None,
        transcript_remove_callback=removed.append,
    )
    pipeline._stop_event.clear()
    pipeline._loop = asyncio.get_running_loop()
    pipeline.transcript_queue = asyncio.Queue()
    orch = asyncio.create_task(pipeline._async_orchestrator())

    phrase_id = uuid.uuid4()
    await pipeline.transcript_queue.put(
        _make_transcript("tha", phrase_id=phrase_id, is_final=False, revision=1)
    )
    await pipeline.transcript_queue.put(
        _make_transcript("", phrase_id=phrase_id, is_final=True, revision=2)
    )
    for _ in range(30):
        if removed:
            break
        await asyncio.sleep(0.01)

    pipeline._stop_event.set()
    orch.cancel()
    try:
        await orch
    except asyncio.CancelledError:
        pass

    assert len(cards) == 1
    assert removed == [cards[0].id]


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
