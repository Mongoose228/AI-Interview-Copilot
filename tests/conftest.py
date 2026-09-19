import time

import pytest

from interview_copilot.models import ProfileSnapshot
from tests.fakes import (
    FakeAudioCapture,
    FakeTranslator,
    FakeWhisperEngine,
)


@pytest.fixture
def fake_audio_capture():
    return FakeAudioCapture()


@pytest.fixture
def fake_translator():
    return FakeTranslator()


@pytest.fixture
def fake_whisper():
    return FakeWhisperEngine()


@pytest.fixture
def sample_profile():
    return ProfileSnapshot(
        name="test",
        content_hash="abc",
        content="Senior Python developer",
        loaded_at=time.time(),
    )
