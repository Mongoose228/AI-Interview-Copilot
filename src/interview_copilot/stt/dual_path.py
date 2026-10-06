"""Speculative dual-path: fast provisional STT + higher-quality final STT."""

from __future__ import annotations

import threading
import time

from ..logging_config import logger
from ..models import SpeechPhrase, Transcript
from .base import SttEngine, TranscriptCallback


class DualPathSttEngine(SttEngine):
    """
    Final engine owns finals (and LLM-bound text). Optional provisional engine
    produces UI-only interim text (tiny Whisper, etc.).
    """

    def __init__(
        self,
        final_engine: SttEngine,
        provisional: SttEngine | None = None,
        *,
        disable_provisional_if_rtf_above: float = 1.0,
        # Backward-compatible aliases
        cloud: SttEngine | None = None,
        local_provisional: SttEngine | None = None,
        disable_local_if_rtf_above: float | None = None,
    ):
        self.final = final_engine if cloud is None else cloud
        self.cloud = self.final  # alias used by older call sites / docs
        self.local = provisional if local_provisional is None else local_provisional
        limit = (
            disable_local_if_rtf_above
            if disable_local_if_rtf_above is not None
            else disable_provisional_if_rtf_above
        )
        self._rtf_limit = limit
        self._local_disabled = False
        self._lock = threading.Lock()
        self._on_transcript: TranscriptCallback | None = None

    @property
    def is_available(self) -> bool:
        return self.final.is_available

    def supports_streaming(self) -> bool:
        return self.final.supports_streaming()

    def warm_up(self) -> None:
        self.final.warm_up()
        if self.local is not None:
            try:
                self.local.warm_up()
            except Exception as e:
                logger.warning(f"Provisional STT warm_up failed: {e}")
                self._local_disabled = True

    def set_keyterms(self, terms: list[str]) -> None:
        self.final.set_keyterms(terms)
        if self.local:
            self.local.set_keyterms(terms)

    def set_initial_prompt(self, prompt: str | None) -> None:
        self.final.set_initial_prompt(prompt)
        if self.local:
            self.local.set_initial_prompt(prompt)

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        return self.final.transcribe(phrase)

    def start_stream(self, on_transcript: TranscriptCallback) -> None:
        self._on_transcript = on_transcript
        self.final.start_stream(on_transcript)

    def feed_audio(self, pcm16: bytes, sample_rate: int = 16000) -> None:
        self.final.feed_audio(pcm16, sample_rate)

    def close_stream(self) -> None:
        self.final.close_stream()
        self._on_transcript = None

    def maybe_provisional(self, phrase: SpeechPhrase) -> Transcript | None:
        """Run provisional STT for UI only. Returns None if disabled."""
        if self.local is None or self._local_disabled or phrase.is_final:
            return None
        t0 = time.time()
        try:
            result = self.local.transcribe(phrase)
        except Exception as e:
            logger.debug(f"Provisional STT failed: {e}")
            return None
        duration = time.time() - t0
        rtf = duration / phrase.duration_s if phrase.duration_s > 0 else 0.0
        if rtf > self._rtf_limit:
            with self._lock:
                self._local_disabled = True
            logger.warning(
                f"Disabling provisional STT (rtf={rtf:.2f} > {self._rtf_limit})"
            )
        return Transcript(
            phrase_id=result.phrase_id,
            text_en=result.text_en,
            language=result.language,
            confidence=result.confidence,
            stt_duration_s=result.stt_duration_s,
            speaker=getattr(phrase, "speaker", result.speaker),
            stt_started_at=result.stt_started_at,
            stt_ended_at=result.stt_ended_at,
            is_final=False,
            revision=phrase.revision,
            speech_started_at=phrase.captured_at,
            speech_ended_at=0.0,
            audio_duration_s=phrase.duration_s,
        )
