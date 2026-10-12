"""STT engine abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ..models import SpeechPhrase, Transcript

TranscriptCallback = Callable[[Transcript], None]


class SttError(RuntimeError):
    """Hard transcription failure (network, HTTP, auth); triggers STT failover."""


class SttEngine(ABC):
    """Common contract for batch and streaming speech-to-text backends."""

    @abstractmethod
    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        """Blocking batch transcription of one phrase."""

    def warm_up(self) -> None:
        return None

    @property
    def is_available(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False

    def start_stream(self, on_transcript: TranscriptCallback) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def feed_audio(self, pcm16: bytes, sample_rate: int = 16000) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def close_stream(self) -> None:
        return None

    def set_keyterms(self, terms: list[str]) -> None:
        """Optional vocabulary boost; default no-op."""
        return None

    def set_initial_prompt(self, prompt: str | None) -> None:
        return None

    def reload_api_key(self) -> bool:
        """Re-read the API key from ``config``; returns True if it changed."""
        return False
