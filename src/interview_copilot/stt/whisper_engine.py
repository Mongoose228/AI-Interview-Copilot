import threading
import time

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
                    dummy_audio, beam_size=1, language="en", condition_on_previous_text=False
                )
                # Consume generator
                list(segments)
                logger.info("Whisper model warmed up.")
            except Exception as e:
                logger.error(f"Failed to warm up Whisper: {e}")

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        """
        Transcribe a SpeechPhrase. Blocks until complete.
        """
        start_time = time.time()

        with self._lock:
            try:
                segments, info = self._model.transcribe(
                    phrase.audio_data,
                    beam_size=5,
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=False,  # We already do VAD
                )

                # Consume generator completely inside the lock to get all text
                texts = []
                for segment in segments:
                    texts.append(segment.text.strip())

                full_text = " ".join(texts).strip()

                stt_duration = time.time() - start_time
                return Transcript(
                    phrase_id=phrase.id,
                    text_en=full_text,
                    language=info.language,
                    confidence=info.language_probability,
                    stt_duration_s=stt_duration,
                )
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                stt_duration = time.time() - start_time
                return Transcript(
                    phrase_id=phrase.id,
                    text_en="",
                    language="en",
                    confidence=0.0,
                    stt_duration_s=stt_duration,
                )
