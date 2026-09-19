"""Shared test doubles for pipeline DI tests."""
import time
import uuid

import numpy as np

from interview_copilot.audio.base import AudioCaptureBackend
from interview_copilot.models import (
    AudioChunk,
    ProfileSnapshot,
    SpeechPhrase,
    SuggestionResult,
    Transcript,
)
from interview_copilot.translation.base import Translator


class FakeAudioCapture(AudioCaptureBackend):
    def __init__(self):
        self._running = False
        self._chunk_count = 0
        self._stopped = False

    def list_devices(self) -> list[dict]:
        return [
            {
                "id": "fake_device",
                "name": "Fake Loopback Device",
                "is_default": True,
                "is_loopback": True,
            }
        ]

    def get_default_loopback(self) -> dict | None:
        return self.list_devices()[0]

    def start(self, device_id: str | None = None) -> None:
        self._running = True
        self._stopped = False
        self._chunk_count = 0

    def read_chunk(self) -> AudioChunk:
        if not self._running:
            raise RuntimeError("Capture not started")
        for _ in range(20):
            if not self._running:
                raise RuntimeError("Capture stopped")
            time.sleep(0.01)
        self._chunk_count += 1
        frames = int(16000 * 0.03)
        data = np.zeros((frames, 2), dtype=np.float32)
        return AudioChunk(
            id=uuid.uuid4(),
            data=data,
            sample_rate=16000,
            channels=2,
            captured_at=time.time(),
        )

    def stop(self) -> None:
        self._running = False
        self._stopped = True

    def is_running(self) -> bool:
        return self._running


class FakeTranslator(Translator):
    def __init__(self, available=True):
        self._available = available

    @property
    def is_available(self) -> bool:
        return self._available

    def translate(self, text: str, source_lang: str = "EN", target_lang: str = "RU") -> str:
        if not text:
            return ""
        return f"[translated_to_{target_lang}] {text}"


class FakeWhisperEngine:
    def __init__(self, fake_text="fake transcription"):
        self.fake_text = fake_text

    def warm_up(self):
        pass

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        started = time.time()
        return Transcript(
            phrase_id=phrase.id,
            text_en=self.fake_text,
            language="en",
            confidence=0.99,
            stt_duration_s=0.1,
            stt_started_at=started,
            stt_ended_at=started + 0.1,
        )


class FakeVAD:
    def __init__(self, available=True):
        self._available = available
        self._flushed = False

    @property
    def is_available(self) -> bool:
        return self._available

    def reset(self):
        pass

    def process_chunk(self, chunk: AudioChunk) -> list[SpeechPhrase]:
        return []

    def flush(self) -> SpeechPhrase | None:
        self._flushed = True
        return None


class FakeSuggester:
    def __init__(self, configured=True, answer="Suggested answer"):
        self._configured = configured
        self.answer = answer
        self.calls = []

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def warm_up(self):
        pass

    async def get_suggestion(self, history, profile, stream_callback=None):
        self.calls.append((history, profile))
        if stream_callback and self.answer:
            await stream_callback(self.answer)
        return SuggestionResult(answer_en=self.answer) if self.answer else None


class FakeProfileManager:
    def __init__(self, profile: ProfileSnapshot | None = None):
        self._profile = profile

    def load_active_profile(self):
        return self._profile

    def load_profile(self, name: str):
        if self._profile and self._profile.name == name:
            return self._profile
        return None

    def list_profiles(self):
        return [self._profile.name] if self._profile else []
