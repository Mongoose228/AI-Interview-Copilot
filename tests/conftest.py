import time
import uuid

import numpy as np
import pytest

from interview_copilot.audio.base import AudioCaptureBackend
from interview_copilot.models import AudioChunk, Transcript
from interview_copilot.translation.base import Translator


class FakeAudioCapture(AudioCaptureBackend):
    def __init__(self):
        self._running = False
        self._chunk_count = 0

    def list_devices(self) -> list[dict]:
        return [{"id": "fake_device", "name": "Fake Loopback Device", "is_default": True}]

    def get_default_loopback(self) -> dict | None:
        return self.list_devices()[0]

    def start(self, device_id: str | None = None) -> None:
        self._running = True
        self._chunk_count = 0

    def read_chunk(self) -> AudioChunk:
        if not self._running:
            raise RuntimeError("Capture not started")

        self._chunk_count += 1
        # generate a fake chunk of 30ms at 48000Hz (1440 samples), stereo, float32
        data = np.zeros((1440, 2), dtype=np.float32).tobytes()
        return AudioChunk(
            id=uuid.uuid4(), data=data, sample_rate=48000, channels=2, captured_at=time.time()
        )

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running


class FakeTranslator(Translator):
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text:
            return ""
        return f"[translated_to_{target_lang}] {text}"


class FakeWhisperEngine:
    def __init__(self, fake_text="fake transcription"):
        self.fake_text = fake_text

    def warm_up(self):
        pass

    def transcribe(self, audio: np.ndarray) -> Transcript:
        return Transcript(
            phrase_id=uuid.uuid4(),
            text_en=self.fake_text,
            language="en",
            confidence=0.99,
            stt_duration_s=0.1,
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
