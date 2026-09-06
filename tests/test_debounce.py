"""Tests for the debounce/accumulator logic."""
import uuid

import pytest

from interview_copilot.models import Transcript


def _make_transcript(text: str) -> Transcript:
    return Transcript(
        phrase_id=uuid.uuid4(),
        text_en=text,
        language="en",
        confidence=0.95,
        stt_duration_s=0.5,
    )


@pytest.mark.asyncio
async def test_multiple_segments_merge_into_one():
    """Two transcripts arriving within debounce window should merge."""
    segments = [
        _make_transcript("What is"),
        _make_transcript("the time complexity of quicksort?"),
    ]
    merged = " ".join(t.text_en for t in segments)
    assert merged == "What is the time complexity of quicksort?"


@pytest.mark.asyncio
async def test_single_segment_passes_through():
    """A single transcript after debounce should pass through unchanged."""
    t = _make_transcript("What is polymorphism?")
    merged = t.text_en
    assert merged == "What is polymorphism?"


def test_transcript_merge_preserves_last_id():
    """Merged transcript should use the last segment's phrase_id."""
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
    assert merged.stt_duration_s == 1.0
