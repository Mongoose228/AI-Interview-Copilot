"""Optional dual capture: system loopback + microphone."""

from __future__ import annotations

import threading
import time
import uuid

import numpy as np

from ..config import config
from ..logging_config import logger
from ..models import AudioChunk
from .base import AudioCaptureBackend
from .soundcard_wasapi import SoundCardWASAPIBackend


class DualAudioCapture(AudioCaptureBackend):
    """
    Primary stream is always loopback (interviewer). Optional mic runs in parallel;
    mic chunks are exposed via read_mic_chunk() for candidate labeling, not LLM by default.
    """

    def __init__(
        self,
        loopback: AudioCaptureBackend | None = None,
        mic: AudioCaptureBackend | None = None,
        *,
        mic_enabled: bool | None = None,
        mic_device_id: str | None = None,
    ):
        self.loopback = loopback or SoundCardWASAPIBackend()
        self.mic = mic or SoundCardWASAPIBackend()
        self._mic_enabled = (
            config.AUDIO_DUAL_CHANNEL if mic_enabled is None else mic_enabled
        )
        self._mic_device_id = (
            mic_device_id if mic_device_id is not None else config.AUDIO_MIC_DEVICE
        )
        self._mic_queue: list[AudioChunk] = []
        self._mic_lock = threading.Lock()
        self._mic_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running = False

    def list_devices(self) -> list[dict]:
        return self.loopback.list_devices()

    def get_default_loopback(self) -> dict | None:
        return self.loopback.get_default_loopback()

    def set_mic_enabled(self, enabled: bool, mic_device_id: str | None = None) -> None:
        self._mic_enabled = enabled
        if mic_device_id is not None:
            self._mic_device_id = mic_device_id

    def _resolve_mic_device_id(self) -> str | None:
        if self._mic_device_id:
            return self._mic_device_id
        try:
            for d in self.loopback.list_devices():
                if not d.get("is_loopback"):
                    return d.get("id")
        except Exception as e:
            logger.debug(f"Could not list mic devices: {e}")
        return None

    def start(self, device_id: str | None = None) -> None:
        self._stop.clear()
        with self._mic_lock:
            self._mic_queue.clear()
        self.loopback.start(device_id)
        self._running = True
        if not self._mic_enabled:
            return
        mic_id = self._resolve_mic_device_id()
        if not mic_id:
            logger.warning(
                "Dual-channel enabled but no microphone device found; "
                "set AUDIO_MIC_DEVICE or pick a mic in Settings"
            )
            return
        try:
            self.mic.start(mic_id)
            self._mic_thread = threading.Thread(
                target=self._mic_loop, name="mic-capture", daemon=True
            )
            self._mic_thread.start()
            logger.info(f"Dual-channel mic capture started ({mic_id})")
        except Exception as e:
            logger.warning(f"Mic capture failed to start: {e}")

    def _mic_loop(self) -> None:
        while not self._stop.is_set() and self.mic.is_running():
            try:
                chunk = self.mic.read_chunk()
            except RuntimeError:
                break
            except Exception as e:
                logger.debug(f"Mic read error: {e}")
                time.sleep(0.05)
                continue
            with self._mic_lock:
                self._mic_queue.append(chunk)
                # Bound memory
                if len(self._mic_queue) > 50:
                    self._mic_queue = self._mic_queue[-20:]

    def read_chunk(self) -> AudioChunk:
        return self.loopback.read_chunk()

    def drain_mic_chunks(self) -> list[AudioChunk]:
        with self._mic_lock:
            chunks = list(self._mic_queue)
            self._mic_queue.clear()
            return chunks

    def stop(self) -> None:
        self._stop.set()
        self._running = False
        try:
            self.loopback.stop()
        except Exception:
            pass
        try:
            self.mic.stop()
        except Exception:
            pass
        if self._mic_thread and self._mic_thread.is_alive():
            self._mic_thread.join(timeout=2.0)
        self._mic_thread = None

    def is_running(self) -> bool:
        return self._running and self.loopback.is_running()


def silence_chunk(sample_rate: int = 16000, duration_ms: int = 20) -> AudioChunk:
    frames = int(sample_rate * (duration_ms / 1000.0))
    return AudioChunk(
        id=uuid.uuid4(),
        data=np.zeros(frames, dtype=np.float32),
        sample_rate=sample_rate,
        channels=1,
        captured_at=time.time(),
    )
