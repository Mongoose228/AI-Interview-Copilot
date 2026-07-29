import numpy as np
import soxr


class Resampler:
    def __init__(self, in_rate: int, out_rate: int = 16000, channels: int = 1):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.channels = channels
        # Silero VAD requires mono 16kHz
        self._resampler = soxr.ResampleStream(in_rate, out_rate, channels, dtype="float32")

    def process(self, audio_data: bytes, in_channels: int = 2) -> np.ndarray:
        """
        Process audio chunk bytes, mixdown to mono if needed, and resample.
        Returns float32 numpy array.
        """
        # Data is float32
        samples = np.frombuffer(audio_data, dtype=np.float32)

        # If stereo, mix to mono
        if in_channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        elif in_channels > 2:
            samples = samples.reshape(-1, in_channels).mean(axis=1)

        # Resample
        resampled = self._resampler.resample_chunk(samples)
        return resampled
