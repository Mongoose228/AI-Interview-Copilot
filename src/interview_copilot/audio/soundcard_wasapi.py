import time
import uuid

import numpy as np
import soundcard as sc

from ..config import config
from ..models import AudioChunk
from .base import AudioCaptureBackend


class SoundCardWASAPIBackend(AudioCaptureBackend):
    def __init__(self):
        self._mic = None
        self._recorder = None
        self._running = False
        self._sample_rate = config.AUDIO_SAMPLE_RATE
        # we will capture loopback, usually 48000Hz default on windows
        # soundcard can capture at its native rate, but we can also just request 48000
        # and then we resample later.
        self._native_rate = 48000
        self._chunk_frames = int(self._native_rate * (config.AUDIO_CHUNK_MS / 1000.0))

    def list_devices(self) -> list[dict]:
        mics = sc.all_microphones(include_loopback=True)
        default_speaker = sc.default_speaker()

        devices = []
        for i, m in enumerate(mics):
            is_loopback = m.isloopback
            # heuristic to find the loopback of default speaker
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
        # first try to find the one marked as default
        for d in devices:
            if d.get("is_default") and d.get("is_loopback"):
                return d
        # if not found, just return first loopback
        for d in devices:
            if d.get("is_loopback"):
                return d
        return None

    def start(self, device_id: str | None = None) -> None:
        if self._running:
            return

        mics = sc.all_microphones(include_loopback=True)
        if device_id:
            for m in mics:
                if str(m.id) == device_id:
                    self._mic = m
                    break
            if not self._mic:
                raise ValueError(f"Device {device_id} not found.")
        else:
            default_info = self.get_default_loopback()
            if not default_info:
                raise RuntimeError("No loopback device found.")
            self._mic = mics[default_info["index"]]

        self._recorder = self._mic.recorder(
            samplerate=self._native_rate, channels=2, blocksize=self._chunk_frames
        )
        self._recorder.__enter__()
        self._running = True

    def read_chunk(self) -> AudioChunk:
        if not self._running or not self._recorder:
            raise RuntimeError("Capture not started")

        # blocks until chunk_frames are available
        data = self._recorder.record(numframes=self._chunk_frames)
        # data is (frames, channels) float32
        data_bytes = data.astype(np.float32).tobytes()

        return AudioChunk(
            id=uuid.uuid4(),
            data=data_bytes,
            sample_rate=self._native_rate,
            channels=2,
            captured_at=time.time(),
        )

    def stop(self) -> None:
        if self._running and self._recorder:
            self._recorder.__exit__(None, None, None)
        self._recorder = None
        self._mic = None
        self._running = False

    def is_running(self) -> bool:
        return self._running
