"""Failover wrapper: try primary STT, fall back on hard failures."""

from __future__ import annotations

import threading
import time

from ..logging_config import logger
from ..models import SpeechPhrase, Transcript
from .base import SttEngine, TranscriptCallback


class FailoverSttEngine(SttEngine):
    # After tripping, batch transcription retries the primary once this cooldown elapses.
    _RETRY_PRIMARY_AFTER_S = 60.0

    def __init__(self, primary: SttEngine, fallback: SttEngine):
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = False
        self._tripped_at = 0.0
        self._lock = threading.Lock()
        self._active_stream: SttEngine | None = None

    @property
    def is_available(self) -> bool:
        return self.primary.is_available or self.fallback.is_available

    def supports_streaming(self) -> bool:
        eng = self._active()
        return eng.supports_streaming()

    def warm_up(self) -> None:
        if self.primary.is_available:
            self.primary.warm_up()
        if self.fallback.is_available:
            self.fallback.warm_up()

    def set_keyterms(self, terms: list[str]) -> None:
        self.primary.set_keyterms(terms)
        self.fallback.set_keyterms(terms)

    def set_initial_prompt(self, prompt: str | None) -> None:
        self.primary.set_initial_prompt(prompt)
        self.fallback.set_initial_prompt(prompt)

    def _active(self) -> SttEngine:
        with self._lock:
            if self._use_fallback or not self.primary.is_available:
                return self.fallback
            return self.primary

    def _trip_to_fallback(self, reason: str) -> None:
        with self._lock:
            self._tripped_at = time.monotonic()
            if self._use_fallback:
                return
            self._use_fallback = True
        logger.warning(f"STT failover engaged: {reason}")

    def _maybe_retry_primary(self) -> None:
        with self._lock:
            if (
                self._use_fallback
                and self._active_stream is None
                and time.monotonic() - self._tripped_at >= self._RETRY_PRIMARY_AFTER_S
            ):
                self._use_fallback = False
                logger.info("STT failover: retrying primary engine")

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        self._maybe_retry_primary()
        eng = self._active()
        try:
            # Empty text on final is not necessarily a hard failure.
            return eng.transcribe(phrase)
        except Exception as e:
            if eng is self.primary and self.fallback.is_available:
                self._trip_to_fallback(str(e))
                return self.fallback.transcribe(phrase)
            raise

    def start_stream(self, on_transcript: TranscriptCallback) -> None:
        eng = self._active()
        if not eng.supports_streaming():
            # Fall through to batch-only mode handled by pipeline.
            raise NotImplementedError("Active STT engine does not support streaming")
        try:
            eng.start_stream(on_transcript)
            self._active_stream = eng
        except Exception as e:
            if eng is self.primary and self.fallback.is_available:
                self._trip_to_fallback(str(e))
                if self.fallback.supports_streaming():
                    self.fallback.start_stream(on_transcript)
                    self._active_stream = self.fallback
                else:
                    raise
            else:
                raise

    def feed_audio(self, pcm16: bytes, sample_rate: int = 16000) -> None:
        eng = self._active_stream or self._active()
        if eng.supports_streaming():
            eng.feed_audio(pcm16, sample_rate)

    def close_stream(self) -> None:
        eng = self._active_stream or self._active()
        try:
            eng.close_stream()
        finally:
            self._active_stream = None
