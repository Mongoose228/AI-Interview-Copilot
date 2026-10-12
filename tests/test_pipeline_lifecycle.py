"""Tests for pipeline stop/start lifecycle with DI fakes."""
import asyncio
import queue
import time
import uuid

import numpy as np
import pytest

from interview_copilot.models import SpeechPhrase
from interview_copilot.pipeline import InterviewPipeline
from tests.fakes import (
    FakeProfileManager,
    FakeSuggester,
    FakeVAD,
    FakeWhisperEngine,
)


@pytest.fixture
def pipeline(fake_audio_capture, sample_profile):
    return InterviewPipeline(
        audio=fake_audio_capture,
        vad=FakeVAD(available=True),
        stt=FakeWhisperEngine(),
        translator=None,
        suggester=FakeSuggester(configured=True),
        profile_mgr=FakeProfileManager(sample_profile),
    )


def test_stop_flushes_vad_and_drains(pipeline, fake_audio_capture):
    vad = FakeVAD(available=True)
    pipeline.vad = vad

    async def run_briefly():
        task = asyncio.create_task(pipeline.start())
        await asyncio.sleep(0.3)
        pipeline.stop()
        try:
            await asyncio.wait_for(task, timeout=3.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    asyncio.run(run_briefly())
    assert vad._flushed
    assert fake_audio_capture._stopped
    assert pipeline._threads == []


class _OnePhraseVAD(FakeVAD):
    """Emits a single final phrase on the first chunk."""

    def __init__(self):
        super().__init__(available=True)
        self._emitted = False

    def process_chunk(self, chunk):
        if self._emitted:
            return []
        self._emitted = True
        return [_phrase(uuid.uuid4(), 1, is_final=True)]


def test_phrase_flows_capture_to_stt_to_llm(fake_audio_capture, sample_profile, monkeypatch):
    monkeypatch.setattr("interview_copilot.pipeline.config.LLM_DEBOUNCE_DELAY_S", 0.01)
    suggester = FakeSuggester(configured=True, answer="An answer")
    p = InterviewPipeline(
        audio=fake_audio_capture,
        vad=_OnePhraseVAD(),
        stt=FakeWhisperEngine(fake_text="What is a closure?"),
        translator=None,
        suggester=suggester,
        profile_mgr=FakeProfileManager(sample_profile),
    )
    suggestions = []
    p.set_callbacks(
        transcript_callback=lambda r: None,
        suggestion_callback=suggestions.append,
        token_callback=lambda *_: None,
        error_callback=lambda *a, **k: None,
        audio_status_callback=lambda *a: None,
    )

    async def run_until_answer():
        task = asyncio.create_task(p.start())
        for _ in range(100):
            if suggestions:
                break
            await asyncio.sleep(0.05)
        p.stop()
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(run_until_answer())

    assert len(suggester.calls) == 1
    assert suggestions[-1].transcript == "What is a closure?"
    assert suggestions[-1].suggestion.answer_en == "An answer"


def test_sentinel_unblocks_stt_worker():
    q = queue.Queue(maxsize=5)
    q.put(None)
    item = q.get(timeout=1.0)
    assert item is None


def test_drop_oldest_on_full_queue():
    q = queue.Queue(maxsize=3)
    q.put("old1")
    q.put("old2")
    q.put("old3")

    new_item = "new_phrase"
    try:
        q.put_nowait(new_item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(new_item)

    items = []
    while not q.empty():
        items.append(q.get_nowait())

    assert items == ["old2", "old3", "new_phrase"]


def _phrase(phrase_id, revision, *, is_final=False):
    return SpeechPhrase(
        id=phrase_id,
        audio_data=np.zeros(16000, dtype=np.float32),
        duration_s=1.0,
        captured_at=time.time() - 1.0,
        vad_end_at=time.time(),
        is_final=is_final,
        revision=revision,
    )


def test_partial_mailbox_keeps_latest(pipeline):
    phrase_id = uuid.uuid4()
    pipeline.queues.put_partial(_phrase(phrase_id, 1))
    pipeline.queues.put_partial(_phrase(phrase_id, 2))

    latest = pipeline.queues.partials.get_nowait()
    assert latest.revision == 2
    assert pipeline.queues.partials.empty()


def test_final_removes_stale_partial_and_has_priority(pipeline):
    phrase_id = uuid.uuid4()
    pipeline.queues.put_partial(_phrase(phrase_id, 1))
    final = _phrase(phrase_id, 2, is_final=True)
    pipeline.queues.put_final(final)

    assert pipeline.queues.partials.empty()
    assert pipeline.queues.finals.get_nowait() is final
    assert phrase_id in pipeline.queues.final_ids


def test_finals_are_served_before_partials(pipeline):
    partial = _phrase(uuid.uuid4(), 1)
    final = _phrase(uuid.uuid4(), 1, is_final=True)
    pipeline.queues.put_partial(partial)
    pipeline.queues.put_final(final)

    assert pipeline.queues.get(timeout=0.1) is final
    assert pipeline.queues.get(timeout=0.1) is partial
    with pytest.raises(queue.Empty):
        pipeline.queues.get(timeout=0.01)


def test_partial_superseded_by_queued_final(pipeline):
    phrase_id = uuid.uuid4()
    final = _phrase(phrase_id, 2, is_final=True)
    pipeline.queues.put_final(final)

    assert pipeline.queues.is_superseded(_phrase(phrase_id, 1))
    pipeline.queues.mark_done(final)
    assert not pipeline.queues.is_superseded(_phrase(phrase_id, 1))
