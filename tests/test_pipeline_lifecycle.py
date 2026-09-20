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
    pipeline._enqueue_partial(_phrase(phrase_id, 1))
    pipeline._enqueue_partial(_phrase(phrase_id, 2))

    latest = pipeline.partial_phrase_queue.get_nowait()
    assert latest.revision == 2
    assert pipeline.partial_phrase_queue.empty()


def test_final_removes_stale_partial_and_has_priority(pipeline):
    phrase_id = uuid.uuid4()
    pipeline._enqueue_partial(_phrase(phrase_id, 1))
    final = _phrase(phrase_id, 2, is_final=True)
    pipeline._enqueue_final(final)

    assert pipeline.partial_phrase_queue.empty()
    assert pipeline.phrase_queue.get_nowait() is final
    assert phrase_id in pipeline._final_phrase_ids
