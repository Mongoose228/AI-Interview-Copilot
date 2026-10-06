"""DSP and STT factory smoke tests."""

import time
import uuid

import numpy as np

from interview_copilot.audio.dsp import (
    float32_to_pcm16,
    process_chunk_audio,
    resample,
    to_mono,
)
from interview_copilot.models import SpeechPhrase
from interview_copilot.stt.deepgram import DeepgramEngine
from interview_copilot.stt.factory import build_stt_engine
from interview_copilot.stt.failover import FailoverSttEngine
from tests.fakes import FakeWhisperEngine


def test_to_mono_energy_picks_louder_channel():
    left = np.ones(100, dtype=np.float32) * 0.01
    right = np.ones(100, dtype=np.float32) * 0.5
    stereo = np.stack([left, right], axis=1)
    mono = to_mono(stereo, mode="energy")
    assert mono.shape == (100,)
    assert float(np.mean(np.abs(mono))) > 0.2


def test_resample_48k_to_16k():
    src = np.sin(np.linspace(0, 10, 48000)).astype(np.float32)
    out = resample(src, 48000, 16000)
    assert 15900 < len(out) < 16100


def test_process_chunk_and_pcm16():
    stereo = np.random.randn(960, 2).astype(np.float32) * 0.1
    mono, sr = process_chunk_audio(stereo, 16000, denoise=False, normalize=True)
    assert sr == 16000
    assert mono.ndim == 1
    pcm = float32_to_pcm16(mono)
    assert len(pcm) == len(mono) * 2


def test_deepgram_unavailable_without_key(monkeypatch):
    monkeypatch.setattr(
        "interview_copilot.stt.deepgram.config.DEEPGRAM_API_KEY", None, raising=False
    )
    eng = DeepgramEngine(api_key=None)
    assert eng.is_available is False
    assert eng.supports_streaming() is True


def test_failover_trips_to_secondary():
    class Boom(FakeWhisperEngine):
        def transcribe(self, phrase):
            raise RuntimeError("primary down")

    primary = Boom()
    fallback = FakeWhisperEngine(fake_text="fallback ok")
    eng = FailoverSttEngine(primary, fallback)

    phrase = SpeechPhrase(
        id=uuid.uuid4(),
        audio_data=np.zeros(1600, dtype=np.float32),
        duration_s=0.1,
        captured_at=time.time(),
        vad_end_at=time.time(),
    )
    result = eng.transcribe(phrase)
    assert result.text_en == "fallback ok"


def test_dual_path_provisional_keeps_speaker(monkeypatch):
    from interview_copilot.stt.dual_path import DualPathSttEngine

    final = FakeWhisperEngine(fake_text="final answer")
    provisional = FakeWhisperEngine(fake_text="fast guess")
    eng = DualPathSttEngine(final_engine=final, provisional=provisional)

    phrase = SpeechPhrase(
        id=uuid.uuid4(),
        audio_data=np.zeros(3200, dtype=np.float32),
        duration_s=0.2,
        captured_at=time.time(),
        vad_end_at=time.time(),
        is_final=False,
        revision=1,
        speaker="candidate",
    )
    prov = eng.maybe_provisional(phrase)
    assert prov is not None
    assert prov.text_en == "fast guess"
    assert prov.is_final is False
    assert prov.speaker == "candidate"

    final_phrase = SpeechPhrase(
        id=phrase.id,
        audio_data=phrase.audio_data,
        duration_s=phrase.duration_s,
        captured_at=phrase.captured_at,
        vad_end_at=phrase.vad_end_at,
        is_final=True,
        revision=2,
        speaker="candidate",
    )
    out = eng.transcribe(final_phrase)
    assert out.text_en == "final answer"
    assert out.speaker == "candidate"


def test_build_stt_engine_wraps_dual_path(monkeypatch):
    from interview_copilot.config import config
    from interview_copilot.stt.dual_path import DualPathSttEngine

    monkeypatch.setattr(config, "STT_BACKEND", "whisper")
    monkeypatch.setattr(config, "STT_FALLBACK", "none")
    monkeypatch.setattr(config, "STT_DUAL_PATH_ENABLED", True)
    monkeypatch.setattr(config, "WHISPER_MODEL", "base.en")
    monkeypatch.setattr(config, "STT_PROVISIONAL_MODEL", "tiny.en")

    # Avoid loading real Whisper weights in unit tests.
    class Tiny(FakeWhisperEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(fake_text="tiny")

    class Base(FakeWhisperEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(fake_text="base")

    monkeypatch.setattr(
        "interview_copilot.stt.factory.WhisperEngine",
        lambda *a, **k: Tiny() if k.get("model_size") == "tiny.en" else Base(),
    )
    engine = build_stt_engine()
    assert isinstance(engine, DualPathSttEngine)


def test_dual_capture_drain_mic():
    from interview_copilot.audio.dual_capture import DualAudioCapture
    from interview_copilot.models import AudioChunk

    class StubBackend:
        def __init__(self):
            self.started = False
            self._chunks = []

        def list_devices(self):
            return [
                {"id": "lb", "name": "Loopback", "is_loopback": True, "is_default": True},
                {"id": "mic1", "name": "Mic", "is_loopback": False, "is_default": False},
            ]

        def get_default_loopback(self):
            return self.list_devices()[0]

        def start(self, device_id=None):
            self.started = True

        def stop(self):
            self.started = False

        def is_running(self):
            return self.started

        def read_chunk(self):
            raise RuntimeError("stop")

    loop = StubBackend()
    mic = StubBackend()
    cap = DualAudioCapture(loopback=loop, mic=mic, mic_enabled=True, mic_device_id="mic1")
    # Inject a chunk as if the mic thread had queued it.
    chunk = AudioChunk(
        id=uuid.uuid4(),
        data=np.zeros(320, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        captured_at=time.time(),
    )
    with cap._mic_lock:
        cap._mic_queue.append(chunk)
    drained = cap.drain_mic_chunks()
    assert len(drained) == 1
    assert drained[0].id == chunk.id
    assert cap.drain_mic_chunks() == []


def test_build_stt_engine_default_whisper():
    from interview_copilot.config import config

    assert config.STT_BACKEND in {"whisper", "deepgram", "groq", "soniox"}
    assert callable(build_stt_engine)
