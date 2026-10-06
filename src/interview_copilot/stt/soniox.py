"""Optional Soniox realtime STT (HTTP finalize path for harness / bake-off)."""

from __future__ import annotations

import time

import httpx
import numpy as np

from ..audio.dsp import float32_to_pcm16
from ..config import config
from ..logging_config import logger
from ..models import SpeechPhrase, Transcript
from .base import SttEngine


class SonioxEngine(SttEngine):
    """
    Batch-style transcription via Soniox REST when SONIOX_API_KEY is set.
    Streaming WebSocket can be added later; harness/bake-off use transcribe().
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key if api_key is not None else config.SONIOX_API_KEY
        self._model = config.SONIOX_MODEL

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def supports_streaming(self) -> bool:
        return False

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        started = time.time()
        if not self.is_available:
            return self._empty(phrase, started)

        audio = np.asarray(phrase.audio_data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        pcm = float32_to_pcm16(audio)
        url = "https://api.soniox.com/v1/stt/transcribe"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/octet-stream",
        }
        params = {
            "model": self._model,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
            "num_channels": 1,
            "language_hints": "en",
        }
        try:
            with httpx.Client(
                timeout=httpx.Timeout(
                    config.NETWORK_READ_TIMEOUT,
                    connect=config.NETWORK_CONNECT_TIMEOUT,
                )
            ) as client:
                resp = client.post(url, params=params, content=pcm, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
            text = self._extract_text(payload)
            ended = time.time()
            return Transcript(
                phrase_id=phrase.id,
                text_en=text,
                language="en",
                confidence=0.0,
                stt_duration_s=ended - started,
                stt_started_at=started,
                stt_ended_at=ended,
                is_final=phrase.is_final,
                revision=phrase.revision,
                speech_started_at=phrase.captured_at,
                speech_ended_at=phrase.vad_end_at,
                audio_duration_s=phrase.duration_s,
            )
        except Exception as e:
            logger.error(f"Soniox transcription failed: {e}")
            return self._empty(phrase, started)

    @staticmethod
    def _extract_text(payload: dict) -> str:
        if isinstance(payload.get("text"), str):
            return payload["text"].strip()
        tokens = payload.get("tokens") or payload.get("words") or []
        parts = []
        for t in tokens:
            if isinstance(t, dict) and t.get("text"):
                parts.append(str(t["text"]))
            elif isinstance(t, str):
                parts.append(t)
        return " ".join(parts).strip()

    def _empty(self, phrase: SpeechPhrase, started: float) -> Transcript:
        return Transcript(
            phrase_id=phrase.id,
            text_en="",
            language="en",
            confidence=0.0,
            stt_duration_s=time.time() - started,
            stt_started_at=started,
            stt_ended_at=time.time(),
            is_final=phrase.is_final,
            revision=phrase.revision,
            speech_started_at=phrase.captured_at,
            speech_ended_at=phrase.vad_end_at,
            audio_duration_s=phrase.duration_s,
        )
