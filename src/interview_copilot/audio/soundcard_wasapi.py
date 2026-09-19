import threading
import time
import uuid
import warnings

import numpy as np
import soundcard as sc

from ..config import config
from ..models import AudioChunk
from .base import AudioCaptureBackend

# Suppress harmless WASAPI discontinuity warnings
warnings.filterwarnings("ignore", message="data discontinuity in recording", module="soundcard")


class SoundCardWASAPIBackend(AudioCaptureBackend):
    def __init__(self):
        self._mic = None
        self._recorder = None
        self._running = False
        self._lock = threading.Lock()
        self._sample_rate = config.AUDIO_SAMPLE_RATE
        self._chunk_frames = int(self._sample_rate * (config.AUDIO_CHUNK_MS / 1000.0))

    def list_devices(self) -> list[dict]:
        mics = sc.all_microphones(include_loopback=True)
        default_speaker = sc.default_speaker()

        devices = []
        for i, m in enumerate(mics):
            is_loopback = m.isloopback
            is_default_loopback = is_loopback and (
                default_speaker.id in m.id
                or m.name in default_speaker.name
                or default_speaker.name in m.name
            )
            devices.append(
                {
                    "id": str(m.id),
                    "name": m.name,
                    "is_default": is_default_loopback,
                    "is_loopback": is_loopback,
                    "index": i,
                }
            )
        return devices

    def get_default_loopback(self) -> dict | None:
        devices = self.list_devices()
        for d in devices:
            if d.get("is_default") and d.get("is_loopback"):
                return d
        for d in devices:
            if d.get("is_loopback"):
                return d
        return None

    def start(self, device_id: str | None = None) -> None:
        with self._lock:
            if self._running:
                return

            mics = sc.all_microphones(include_loopback=True)
            if device_id:
                for m in mics:
                    if str(m.id) == device_id:
                        self._mic = m
                        if not m.isloopback:
                            from ..logging_config import logger
                            logger.warning(
                                f"Selected device '{m.name}' is NOT a loopback device. "
                                f"Audio will be captured from the microphone, not system output."
                            )
                        break
                if not self._mic:
                    raise ValueError(f"Device {device_id} not found.")
            else:
                default_info = self.get_default_loopback()
                if not default_info:
                    raise RuntimeError("No loopback device found.")
                self._mic = mics[default_info["index"]]

            self._recorder = self._mic.recorder(
                samplerate=self._sample_rate, channels=2, blocksize=self._chunk_frames
            )
            self._recorder.__enter__()
            self._running = True

    def read_chunk(self) -> AudioChunk:
        with self._lock:
            if not self._running or not self._recorder:
                raise RuntimeError("Capture not started")
            recorder = self._recorder
            sample_rate = self._sample_rate
            chunk_frames = self._chunk_frames

        # record() blocks; do not hold the lock so stop() can interrupt
        try:
            data = recorder.record(numframes=chunk_frames)
        except Exception:
            if not self._running:
                raise RuntimeError("Capture stopped") from None
            raise

        return AudioChunk(
            id=uuid.uuid4(),
            data=data.astype(np.float32),
            sample_rate=sample_rate,
            channels=2,
            captured_at=time.time(),
        )

    def stop(self) -> None:
        with self._lock:
            if self._running and self._recorder:
                try:
                    self._recorder.__exit__(None, None, None)
                except Exception:
                    pass
            self._recorder = None
            self._mic = None
            self._running = False

    def is_running(self) -> bool:
        return self._running
