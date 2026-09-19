import hashlib
import shutil
import time
import urllib.error
import urllib.request
import uuid
from collections import deque

import numpy as np

from ..config import config
from ..logging_config import logger
from ..models import AudioChunk, SpeechPhrase
from ..paths import get_models_dir

# Expected sha256 of silero_vad.onnx (v4.0)
_ONNX_SHA256 = "5605d3c01c0cf07ed9f30325fddb73248c8b4aebda3ec39cb16eebc89d280eec"

# Pre-roll buffer size
_PREROLL_CHUNKS = max(1, int(config.VAD_SPEECH_PAD_MS / (512 / 16.0)))

# ONNX model URL (v4 - compatible with direct onnxruntime usage)
_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"


class SileroVAD:
    def __init__(self):
        self._vad_lib_available = False
        self._session = None
        self._h = None
        self._c = None

        try:
            import onnxruntime as ort

            # Download ONNX if missing (into data-dir/models, not context/)
            model_path = get_models_dir() / "silero_vad.onnx"
            if not model_path.exists():
                logger.info(f"Downloading Silero VAD model to {model_path}...")
                model_path.parent.mkdir(parents=True, exist_ok=True)

                tmp_path = model_path.with_suffix(".tmp")
                try:
                    with (
                        urllib.request.urlopen(_ONNX_URL, timeout=30) as response,
                        open(tmp_path, 'wb') as out_file,
                    ):
                        shutil.copyfileobj(response, out_file)

                    # Verify sha256
                    with open(tmp_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    if file_hash != _ONNX_SHA256:
                        raise ValueError(
                            f"Checksum mismatch: expected {_ONNX_SHA256},"
                            f" got {file_hash}"
                        )

                    tmp_path.replace(model_path)
                    logger.info("Download complete.")
                except Exception as e:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    raise e

            # Initialize InferenceSession
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            # Run on CPU only
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            # Detect model version by checking input names
            input_names = [inp.name for inp in self._session.get_inputs()]
            if 'h' in input_names and 'c' in input_names:
                # v4 model: separate h and c states, shape (2, 1, 64)
                self._model_version = 4
                logger.info("Detected Silero VAD v4 ONNX model.")
            else:
                # v5 model: combined state, shape (2, 1, 128)
                self._model_version = 5
                logger.info("Detected Silero VAD v5 ONNX model.")

            self._reset_onnx_state()
            self._vad_lib_available = True
        except Exception as e:
            logger.error(f"Failed to initialize ONNX VAD: {e}")

        self._sample_rate = 16000

        # State for phrase buffering
        self._phrase_buffer = []
        self._phrase_duration_ms = 0
        self._silence_duration_ms = 0
        self._is_speaking = False
        self._current_phrase_start_time = 0.0
        self._sample_buffer = np.array([], dtype=np.float32)

        # Pre-roll ring buffer: keeps last N chunks so we don't clip speech onset
        self._preroll_buffer: deque[np.ndarray] = deque(maxlen=_PREROLL_CHUNKS)

    @property
    def is_available(self) -> bool:
        """Return True if VAD model loaded successfully."""
        return self._vad_lib_available

    def _reset_onnx_state(self):
        if self._model_version == 4:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 128), dtype=np.float32)
            self._c = None  # v5 uses combined state

    def _infer(self, chunk: np.ndarray) -> float:
        if not self._session:
            return 0.0
        # Expected shape: (1, 512)
        x = chunk.reshape(1, 512).astype(np.float32)
        sr = np.array(self._sample_rate, dtype=np.int64)

        if self._model_version == 4:
            ort_inputs = {
                'input': x,
                'h': self._h,
                'c': self._c,
                'sr': sr
            }
            ort_outs = self._session.run(None, ort_inputs)
            out, self._h, self._c = ort_outs[0], ort_outs[1], ort_outs[2]
        else:
            ort_inputs = {
                'input': x,
                'state': self._h,
                'sr': sr
            }
            ort_outs = self._session.run(None, ort_inputs)
            out, self._h = ort_outs[0], ort_outs[1]

        return float(out[0][0])

    def reset(self):
        """Reset VAD internal state. Call on capture start/stop to prevent drift."""
        if self._vad_lib_available:
            self._reset_onnx_state()
        self._phrase_buffer = []
        self._phrase_duration_ms = 0
        self._silence_duration_ms = 0
        self._is_speaking = False
        self._current_phrase_start_time = 0.0
        self._sample_buffer = np.array([], dtype=np.float32)
        self._preroll_buffer.clear()

    def process_chunk(self, chunk: AudioChunk) -> list[SpeechPhrase]:
        """
        Process a chunk of audio, returning any completed phrases.
        """
        if not self._vad_lib_available:
            return []

        # We assume the input is already 16kHz mono (e.g. from WASAPI or soundcard config)
        # However, WASAPI returns stereo by default, so we mix to mono if needed
        data = chunk.data
        if data.ndim == 2 and data.shape[1] > 1:
            data = np.mean(data, axis=1)

        # Append to sample buffer
        self._sample_buffer = np.concatenate([self._sample_buffer, data.astype(np.float32)])

        phrases = []
        # Process in chunks of exactly 512 samples
        while len(self._sample_buffer) >= 512:
            vad_chunk = self._sample_buffer[:512]
            self._sample_buffer = self._sample_buffer[512:]

            try:
                prob = self._infer(vad_chunk)

                if prob >= config.VAD_THRESHOLD:
                    self._silence_duration_ms = 0
                    if not self._is_speaking:
                        # START SPEAKING
                        self._is_speaking = True
                        self._phrase_buffer = list(self._preroll_buffer)
                        self._preroll_buffer.clear()
                        self._phrase_buffer.append(vad_chunk)
                        self._current_phrase_start_time = chunk.captured_at
                        self._phrase_duration_ms = len(self._phrase_buffer) * (512 / 16.0)
                    else:
                        self._phrase_buffer.append(vad_chunk)
                        self._phrase_duration_ms += 512 / 16.0
                else:
                    if self._is_speaking:
                        self._silence_duration_ms += 512 / 16.0
                        self._phrase_buffer.append(vad_chunk)
                        self._phrase_duration_ms += 512 / 16.0

                        if self._silence_duration_ms >= config.VAD_SILENCE_MS:
                            # END SPEAKING
                            self._is_speaking = False

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
                            self._silence_duration_ms = 0
                    else:
                        # Not speaking: feed pre-roll buffer
                        self._preroll_buffer.append(vad_chunk)

                # Check max phrase length to force split
                if (
                    self._is_speaking
                    and self._phrase_duration_ms
                    > config.VAD_MAX_PHRASE_SECONDS * 1000
                ):
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
                    self._silence_duration_ms = 0
                    self._current_phrase_start_time = chunk.captured_at

            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logger.error(f"VAD Error: {e}")

        return phrases

    def flush(self) -> SpeechPhrase | None:
        """Flush the current phrase buffer, returning the phrase if one was being recorded."""
        if not self._vad_lib_available:
            return None

        if self._phrase_buffer and self._phrase_duration_ms > config.VAD_MIN_SPEECH_MS:
            full_phrase_audio = np.concatenate(self._phrase_buffer)
            duration_s = len(full_phrase_audio) / self._sample_rate
            phrase = SpeechPhrase(
                id=uuid.uuid4(),
                audio_data=full_phrase_audio,
                duration_s=duration_s,
                captured_at=self._current_phrase_start_time,
                vad_end_at=time.time(),
            )
            self._phrase_buffer = []
            self._phrase_duration_ms = 0
            self._silence_duration_ms = 0
            self._is_speaking = False
            return phrase

        self._phrase_buffer = []
        self._phrase_duration_ms = 0
        self._silence_duration_ms = 0
        self._is_speaking = False
        return None
