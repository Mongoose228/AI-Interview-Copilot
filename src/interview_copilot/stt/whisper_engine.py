import threading
import time
import math

import numpy as np
from faster_whisper import WhisperModel

from ..config import config
from ..logging_config import logger
from ..models import SpeechPhrase, Transcript


class WhisperEngine:
    def __init__(self):
        self._model_size = config.WHISPER_MODEL
        self._device = config.WHISPER_DEVICE
        self._compute_type = config.WHISPER_COMPUTE_TYPE

        # Resolve device/compute type based on 'auto' setting
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
            f"Initializing WhisperModel '{self._model_size}' on {self._device} ({self._compute_type})"
        )
        print(f"Downloading/Loading Whisper model '{self._model_size}', please wait...")

        # Load model. If it's not present, faster-whisper will download it automatically to the cache.
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            local_files_only=False,  # Allows auto-download
            cpu_threads=config.WHISPER_CPU_THREADS,
        )
        print("Whisper model loaded successfully.")

        # Semaphore to ensure only 1 transcription at a time
        # (Though ThreadPoolExecutor handles it, it's good practice inside the engine if called from outside)
        self._lock = threading.Lock()

        # Warmup
        self.warm_up()

    def warm_up(self):
        """Run a dummy audio through the model to JIT compile and load weights into VRAM/RAM."""
        logger.info("Warming up Whisper model...")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        with self._lock:
            try:
                segments, _ = self._model.transcribe(
                    dummy_audio, 
                    beam_size=config.WHISPER_BEAM_SIZE, 
                    language="en", 
                    condition_on_previous_text=False,
                    temperature=0.0,
                    without_timestamps=True
                )
                # Consume generator
                list(segments)
                logger.info("Whisper model warmed up.")
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logger.error(f"Failed to warm up Whisper: {e}")

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        """
        Transcribe a SpeechPhrase. Blocks until complete.
        """
        start_time = time.time()

        with self._lock:
            try:
                segments_gen, info = self._model.transcribe(
                    phrase.audio_data,
                    beam_size=config.WHISPER_BEAM_SIZE,
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=False,  # We already do VAD
                    temperature=0.0,
                    without_timestamps=True,
                )

                # Consume generator completely inside the lock
                segments = list(segments_gen)
                stt_duration = time.time() - start_time

                if not segments:
                    return Transcript(phrase.id, "", "en", 0.0, stt_duration)

                avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
                max_no_speech = max(s.no_speech_prob for s in segments)

                # 1. Filter by metrics
                if avg_logprob < -1.0 or max_no_speech > 0.6:
                    logger.debug(f"Dropped hallucination by metrics: logprob={avg_logprob:.2f}, no_speech={max_no_speech:.2f}")
                    return Transcript(phrase.id, "", "en", 0.0, stt_duration)

                texts = [s.text.strip() for s in segments]
                full_text = " ".join(texts).strip()

                # 2. Filter by blacklist
                text_lower = full_text.lower()
                blacklist = {
                    "thank you.", "thanks for watching!", "bye.", "you.", "...", 
                    "okay.", "so.", "thank you", "thanks for watching", "bye", 
                    "you", "okay", "so"
                }
                if text_lower in blacklist or not text_lower:
                    logger.debug(f"Dropped hallucination by blacklist: '{full_text}'")
                    return Transcript(phrase.id, "", "en", 0.0, stt_duration)

                # Use exp(avg_logprob) as a more meaningful confidence metric than language_probability
                confidence = math.exp(avg_logprob)

                return Transcript(
                    phrase_id=phrase.id,
                    text_en=full_text,
                    language=info.language,
                    confidence=confidence,
                    stt_duration_s=stt_duration,
                )
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logger.error(f"Whisper transcription failed: {e}")
                stt_duration = time.time() - start_time
                return Transcript(
                    phrase_id=phrase.id,
                    text_en="",
                    language="en",
                    confidence=0.0,
                    stt_duration_s=stt_duration,
                )
