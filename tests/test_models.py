import uuid
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from interview_copilot.models import (
    AudioChunk,
    PipelineResult,
)


def test_audio_chunk_immutability():
    chunk = AudioChunk(
        id=uuid.uuid4(),
        data=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        captured_at=123.45,
    )

    assert chunk.sample_rate == 16000
    with pytest.raises(FrozenInstanceError):
        chunk.sample_rate = 48000


def test_pipeline_result_creation():
    result = PipelineResult(
        id=uuid.uuid4(),
        transcript="Hello world",
        translation_ru="Привет мир",
        suggestion=None,
        profile=None,
        timings=[],
        errors=[],
        created_at=1234.5,
    )

    assert result.transcript == "Hello world"
    assert len(result.timings) == 0

    # ensure it's immutable
    with pytest.raises(FrozenInstanceError):
        result.transcript = "Changed"
