"""Groq Whisper (OpenAI-compatible) batch STT."""

from __future__ import annotations

import io
import time
import wave

import numpy as np
from openai import OpenAI, OpenAIError

from ..config import config
from ..logging_config import logger
from ..models import SpeechPhrase, Transcript
from .base import SttEngine, SttError


class GroqWhisperEngine(SttEngine):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key if api_key is not None else config.GROQ_API_KEY
        self._model = model or config.GROQ_WHISPER_MODEL
        self._prompt: str | None = None
        self._client: OpenAI | None = None

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def set_initial_prompt(self, prompt: str | None) -> None:
        self._prompt = prompt

    def reload_api_key(self) -> bool:
        new_key = config.GROQ_API_KEY
        if new_key == self._api_key:
            return False
        self._api_key = new_key
        self._client = None
        return True

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=config.GROQ_BASE_URL,
                timeout=config.NETWORK_READ_TIMEOUT,
            )
        return self._client

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        started = time.time()
        if not self.is_available:
            return self._empty(phrase, started)

        wav_bytes = self._to_wav_bytes(phrase.audio_data)
        kwargs = {
            "model": self._model,
            "file": ("audio.wav", wav_bytes, "audio/wav"),
            "language": "en",
            "response_format": "verbose_json",
        }
        if self._prompt:
            kwargs["prompt"] = self._prompt
        try:
            result = self._get_client().audio.transcriptions.create(**kwargs)
        except OpenAIError as e:
            logger.error(f"Groq Whisper failed: {e}")
            raise SttError(f"Groq Whisper failed: {e}") from e
        text = (getattr(result, "text", None) or "").strip()
        ended = time.time()
        return Transcript(
            phrase_id=phrase.id,
            text_en=text,
            language="en",
            confidence=0.0,
            stt_duration_s=ended - started,
            speaker=getattr(phrase, "speaker", "interviewer"),
            stt_started_at=started,
            stt_ended_at=ended,
            is_final=phrase.is_final,
            revision=phrase.revision,
            speech_started_at=phrase.captured_at,
            speech_ended_at=phrase.vad_end_at,
            audio_duration_s=phrase.duration_s,
        )

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray) -> bytes:
        data = np.asarray(audio, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        pcm = np.clip(data, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def _empty(self, phrase: SpeechPhrase, started: float) -> Transcript:
        return Transcript(
            phrase_id=phrase.id,
            text_en="",
            language="en",
            confidence=0.0,
            stt_duration_s=time.time() - started,
            speaker=getattr(phrase, "speaker", "interviewer"),
            stt_started_at=started,
            stt_ended_at=time.time(),
            is_final=phrase.is_final,
            revision=phrase.revision,
            speech_started_at=phrase.captured_at,
            speech_ended_at=phrase.vad_end_at,
            audio_duration_s=phrase.duration_s,
        )
