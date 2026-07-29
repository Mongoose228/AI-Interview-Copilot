from abc import ABC, abstractmethod

from ..models import AudioChunk


class AudioCaptureBackend(ABC):
    @abstractmethod
    def list_devices(self) -> list[dict]:
        """Return a list of available loopback devices."""

    @abstractmethod
    def get_default_loopback(self) -> dict | None:
        """Find the loopback device corresponding to the default speaker."""

    @abstractmethod
    def start(self, device_id: str | None = None) -> None:
        """Start capturing audio from the specified device (or default)."""

    @abstractmethod
    def read_chunk(self) -> AudioChunk:
        """Read a chunk of audio data. Blocks until data is available."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and release resources."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return True if capturing is currently active."""
