import uuid
from collections import deque

import numpy as np

from ..config import config
from ..logging_config import logger
from ..models import AudioChunk, SpeechPhrase
from .resampler import Resampler

# Pre-roll buffer size: ~500ms at 16kHz in 512-sample chunks ≈ 15 chunks
_PREROLL_CHUNKS = 15


class SileroVAD:
    def __init__(self):
        try:
            from silero_vad import load_silero_vad
            from silero_vad.utils_vad import VADIterator

            self._model = load_silero_vad(onnx=True)
            self._iterator = VADIterator(
                self._model,
                threshold=config.VAD_THRESHOLD,
                sampling_rate=16000,
                min_silence_duration_ms=config.VAD_SILENCE_MS,
                speech_pad_ms=config.VAD_SPEECH_PAD_MS,
            )
            self._vad_lib_available = True
        except ImportError:
            logger.error("silero_vad package not found or incompatible.")
            self._vad_lib_available = False

        self._sample_rate = 16000
        self._resampler = None

        # State for phrase buffering
        self._phrase_buffer = []
        self._phrase_duration_ms = 0
        self._is_speaking = False
        self._current_phrase_start_time = 0.0
        self._sample_buffer = np.array([], dtype=np.float32)

        # Pre-roll ring buffer: keeps last N chunks so we don't clip speech onset
        self._preroll_buffer: deque[np.ndarray] = deque(maxlen=_PREROLL_CHUNKS)

    def reset(self):
        """Reset VAD internal state. Call on capture start/stop to prevent drift."""
        if self._vad_lib_available:
            self._iterator.reset_states()
        self._phrase_buffer = []
        self._phrase_duration_ms = 0
        self._is_speaking = False
        self._current_phrase_start_time = 0.0
        self._sample_buffer = np.array([], dtype=np.float32)
        self._preroll_buffer.clear()

    def _ensure_resampler(self, in_rate: int, in_channels: int):
        if self._resampler is None or self._resampler.in_rate != in_rate:
            self._resampler = Resampler(in_rate, self._sample_rate, 1)

    def process_chunk(self, chunk: AudioChunk) -> list[SpeechPhrase]:
        """
        Process a chunk of audio, returning any completed phrases.
        """
        if not self._vad_lib_available:
            return []

        self._ensure_resampler(chunk.sample_rate, chunk.channels)

        mono_16k = self._resampler.process(chunk.data, chunk.channels)

        # Append to sample buffer
        self._sample_buffer = np.concatenate([self._sample_buffer, mono_16k])

        phrases = []
        import torch

        # Process in chunks of exactly 512 samples
        while len(self._sample_buffer) >= 512:
            vad_chunk = self._sample_buffer[:512]
            self._sample_buffer = self._sample_buffer[512:]

            try:
                tensor = torch.from_numpy(vad_chunk)
                result = self._iterator(tensor, return_seconds=True)

                # Simple buffering logic based on VAD state
                if result:
                    if "start" in result:
                        self._is_speaking = True
                        # Include pre-roll buffer to avoid clipping speech onset
                        self._phrase_buffer = list(self._preroll_buffer)
                        self._phrase_buffer.append(vad_chunk)
                        self._current_phrase_start_time = chunk.captured_at
                        self._phrase_duration_ms = len(self._phrase_buffer) * (512 / 16.0)
                    elif "end" in result:
                        self._is_speaking = False
                        self._phrase_buffer.append(vad_chunk)

                        full_phrase_audio = np.concatenate(self._phrase_buffer)
                        duration_s = len(full_phrase_audio) / self._sample_rate

                        if duration_s * 1000 >= config.VAD_MIN_SPEECH_MS:
                            phrases.append(
                                SpeechPhrase(
                                    id=uuid.uuid4(),
                                    audio_data=full_phrase_audio,
                                    duration_s=duration_s,
                                    captured_at=self._current_phrase_start_time,
                                    vad_end_at=chunk.captured_at,
                                )
                            )

                        self._phrase_buffer = []
                        self._phrase_duration_ms = 0
                else:
                    if self._is_speaking:
                        self._phrase_buffer.append(vad_chunk)
                        self._phrase_duration_ms += 512 / 16.0

                        # Check max phrase length to force split
                        if self._phrase_duration_ms > config.VAD_MAX_PHRASE_SECONDS * 1000:
                            full_phrase_audio = np.concatenate(self._phrase_buffer)
                            phrases.append(
                                SpeechPhrase(
                                    id=uuid.uuid4(),
                                    audio_data=full_phrase_audio,
                                    duration_s=len(full_phrase_audio) / self._sample_rate,
                                    captured_at=self._current_phrase_start_time,
                                    vad_end_at=chunk.captured_at,
                                )
                            )
                            # Reset but keep speaking true to continue buffering next part
                            self._phrase_buffer = []
                            self._phrase_duration_ms = 0
                            self._current_phrase_start_time = chunk.captured_at
                    else:
                        # Not speaking: feed pre-roll buffer
                        self._preroll_buffer.append(vad_chunk)

            except Exception as e:
                logger.error(f"VAD Error: {e}")

        return phrases
