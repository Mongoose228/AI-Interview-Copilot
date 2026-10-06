import math
import threading
import time

import numpy as np
from faster_whisper import WhisperModel

from ..config import config
from ..logging_config import logger
from ..models import SpeechPhrase, Transcript
from .base import SttEngine


class WhisperEngine(SttEngine):
    def __init__(
        self,
        model_size: str | None = None,
        *,
        skip_if_unavailable: bool = False,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        self._model_size = model_size or config.WHISPER_MODEL
        self._device = device or config.WHISPER_DEVICE
        self._compute_type = compute_type or config.WHISPER_COMPUTE_TYPE
        self._initial_prompt: str | None = None
        self._model = None
        self._lock = threading.Lock()

        if self._device == "auto":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    self._device = "cuda"
                else:
                    self._device = "cpu"
            except ImportError:
                self._device = "cpu"

        if self._compute_type == "auto":
            self._compute_type = "float16" if self._device == "cuda" else "int8"

        logger.info(
            f"Initializing WhisperModel '{self._model_size}'"
            f" on {self._device} ({self._compute_type})"
        )
        logger.info(
            f"[STT Config] model={self._model_size} device={self._device} "
            f"compute={self._compute_type} cpu_threads={config.WHISPER_CPU_THREADS}"
        )
        try:
            logger.info(
                f"Downloading/Loading Whisper model '{self._model_size}', please wait..."
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                local_files_only=False,
                cpu_threads=config.WHISPER_CPU_THREADS,
            )
            logger.info("Whisper model loaded successfully.")
            self.warm_up()
        except Exception as e:
            if skip_if_unavailable:
                logger.warning(f"WhisperEngine unavailable ({self._model_size}): {e}")
                self._model = None
            else:
                raise

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def set_initial_prompt(self, prompt: str | None) -> None:
        self._initial_prompt = prompt

    def warm_up(self):
        if self._model is None:
            return
        logger.info("Warming up Whisper model...")
        dummy_audio = np.zeros(16000, dtype=np.float32)
        with self._lock:
            try:
                segments, _ = self._model.transcribe(
                    dummy_audio,
                    beam_size=config.WHISPER_BEAM_SIZE,
                    language="en",
                    condition_on_previous_text=False,
                    temperature=0.0,
                    without_timestamps=True,
                    initial_prompt=self._initial_prompt,
                )
                list(segments)
                logger.info("Whisper model warmed up.")
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logger.error(f"Failed to warm up Whisper: {e}")

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        start_time = time.time()
        speaker = getattr(phrase, "speaker", "interviewer")

        def _blank(stt_duration: float = 0.0) -> Transcript:
            return Transcript(
                phrase_id=phrase.id,
                text_en="",
                language="en",
                confidence=0.0,
                stt_duration_s=stt_duration,
                speaker=speaker,
                stt_started_at=start_time,
                stt_ended_at=time.time(),
                is_final=phrase.is_final,
                revision=phrase.revision,
                speech_started_at=phrase.captured_at,
                speech_ended_at=phrase.vad_end_at,
                audio_duration_s=phrase.duration_s,
            )

        if self._model is None:
            return _blank(0.0)

        with self._lock:
            try:
                segments_gen, info = self._model.transcribe(
                    phrase.audio_data,
                    beam_size=config.WHISPER_BEAM_SIZE,
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=False,
                    temperature=0.0,
                    without_timestamps=True,
                    initial_prompt=self._initial_prompt,
                )

                segments = list(segments_gen)
                stt_duration = time.time() - start_time
                rtf = stt_duration / phrase.duration_s if phrase.duration_s > 0 else 0.0
                logger.info(
                    f"[STT] final={phrase.is_final} revision={phrase.revision} "
                    f"audio={phrase.duration_s:.3f}s duration={stt_duration:.3f}s "
                    f"rtf={rtf:.3f} speaker={speaker}"
                )

                if not segments:
                    return _blank(stt_duration)

                avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
                max_no_speech = max(s.no_speech_prob for s in segments)

                if avg_logprob < -1.0 or max_no_speech > 0.6:
                    logger.debug(
                        f"Dropped hallucination by metrics:"
                        f" logprob={avg_logprob:.2f},"
                        f" no_speech={max_no_speech:.2f}"
                    )
                    return _blank(stt_duration)

                texts = [s.text.strip() for s in segments]
                full_text = " ".join(texts).strip()

                text_lower = full_text.lower()
                blacklist = {
                    "thank you.",
                    "thanks for watching!",
                    "bye.",
                    "you.",
                    "...",
                    "okay.",
                    "so.",
                    "thank you",
                    "thanks for watching",
                    "bye",
                    "you",
                    "okay",
                    "so",
                }
                if text_lower in blacklist or not text_lower:
                    logger.debug(f"Dropped hallucination by blacklist: '{full_text}'")
                    return _blank(stt_duration)

                confidence = math.exp(avg_logprob)
                ended = time.time()

                return Transcript(
                    phrase_id=phrase.id,
                    text_en=full_text,
                    language=info.language,
                    confidence=confidence,
                    stt_duration_s=stt_duration,
                    speaker=speaker,
                    stt_started_at=start_time,
                    stt_ended_at=ended,
                    is_final=phrase.is_final,
                    revision=phrase.revision,
                    speech_started_at=phrase.captured_at,
                    speech_ended_at=phrase.vad_end_at,
                    audio_duration_s=phrase.duration_s,
                )
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logger.error(f"Whisper transcription failed: {e}")
                return _blank(time.time() - start_time)
