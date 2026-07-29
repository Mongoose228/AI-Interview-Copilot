from ..models import AudioChunk
from .base import AudioCaptureBackend


class PyAudioWPatchBackend(AudioCaptureBackend):
    """Fallback backend using pyaudiowpatch if SoundCard fails."""

    def __init__(self):
        self._running = False
        raise NotImplementedError(
            "PyAudioWPatch is not fully implemented. Using SoundCard instead."
        )

    def list_devices(self) -> list[dict]:
        return []

    def get_default_loopback(self) -> dict | None:
        return None

    def start(self, device_id: str | None = None) -> None:
        pass

    def read_chunk(self) -> AudioChunk:
        raise NotImplementedError()

    def stop(self) -> None:
        pass

    def is_running(self) -> bool:
        return self._running
