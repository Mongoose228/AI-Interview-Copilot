"""Audio DSP: mono mix, resample, normalize, light denoise."""

from __future__ import annotations

import numpy as np

from ..config import config
from ..logging_config import logger

_TARGET_SR = 16000


def to_mono(data: np.ndarray, mode: str = "energy") -> np.ndarray:
    """Convert to mono float32. mode: energy | mean | left | right."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        return arr.reshape(-1).astype(np.float32)
    channels = arr.shape[1]
    if channels == 1:
        return arr[:, 0]
    mode = (mode or "energy").lower()
    if mode == "left":
        return arr[:, 0]
    if mode == "right":
        return arr[:, min(1, channels - 1)]
    if mode == "mean":
        return np.mean(arr, axis=1).astype(np.float32)
    # energy: pick louder channel
    energies = np.mean(np.square(arr), axis=0)
    return arr[:, int(np.argmax(energies))].astype(np.float32)


def resample(data: np.ndarray, src_sr: int, dst_sr: int = _TARGET_SR) -> np.ndarray:
    if src_sr == dst_sr or data.size == 0:
        return data.astype(np.float32)
    try:
        import soxr

        return soxr.resample(data.astype(np.float32), src_sr, dst_sr).astype(np.float32)
    except Exception as e:
        logger.warning(f"soxr resample failed ({e}); falling back to linear")
        duration = len(data) / float(src_sr)
        n_out = max(1, int(round(duration * dst_sr)))
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, data.astype(np.float32)).astype(np.float32)


def normalize_peak(data: np.ndarray, peak: float = 0.9) -> np.ndarray:
    if data.size == 0:
        return data
    max_abs = float(np.max(np.abs(data)))
    if max_abs < 1e-6:
        return data
    gain = min(peak / max_abs, 8.0)  # soft cap
    out = data * gain
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def light_denoise(data: np.ndarray, sample_rate: int = _TARGET_SR) -> np.ndarray:
    """Very light spectral gate; skips if disabled or clip too short."""
    if not getattr(config, "DSP_DENOISE_ENABLED", True):
        return data
    if data.size < sample_rate // 10:
        return data
    # Noise floor from quietest 10% frames (20 ms)
    frame = max(1, int(sample_rate * 0.02))
    n_frames = len(data) // frame
    if n_frames < 4:
        return data
    frames = data[: n_frames * frame].reshape(n_frames, frame)
    energies = np.mean(np.square(frames), axis=1)
    noise = float(np.percentile(energies, 10))
    thresh = noise * 4.0
    mask = energies >= thresh
    # Smooth mask
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(mask.astype(np.float32), kernel, mode="same") > 0.3
    gated = frames * smooth[:, None]
    out = gated.reshape(-1)
    if len(out) < len(data):
        out = np.concatenate([out, data[len(out) :]])
    return out.astype(np.float32)


def _apply_gain_ramp(data: np.ndarray, start: float, end: float) -> np.ndarray:
    if start == end:
        out = data * end
    else:
        out = data * np.linspace(start, end, num=len(data), dtype=np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


class AutoGain:
    """Smoothed peak AGC for a stream of short chunks (fast attack, slow release)."""

    def __init__(
        self,
        target_peak: float = 0.9,
        max_gain: float = 8.0,
        noise_floor: float = 0.01,
        attack: float = 0.5,
        release: float = 0.03,
    ):
        self.target_peak = target_peak
        self.max_gain = max_gain
        self.noise_floor = noise_floor
        self.attack = attack
        self.release = release
        self.gain = 1.0

    def reset(self) -> None:
        self.gain = 1.0

    def process(self, data: np.ndarray) -> np.ndarray:
        if data.size == 0:
            return data
        peak = float(np.max(np.abs(data)))
        desired = min(self.target_peak / max(peak, 1e-6), self.max_gain)
        if peak < self.noise_floor:
            # Never boost near-silence; only allow the gain to come down.
            desired = min(desired, self.gain)
        coeff = self.attack if desired < self.gain else self.release
        prev = self.gain
        self.gain = prev + (desired - prev) * coeff
        return _apply_gain_ramp(data, prev, self.gain)


class NoiseGate:
    """Light gate against a running noise-floor estimate kept across chunks."""

    def __init__(
        self,
        open_ratio: float = 4.0,
        attenuation: float = 0.1,
        hold_chunks: int = 10,
        floor_fall: float = 0.2,
        floor_rise: float = 0.002,
    ):
        self.open_ratio = open_ratio
        self.attenuation = attenuation
        self.hold_chunks = hold_chunks
        self.floor_fall = floor_fall
        self.floor_rise = floor_rise
        self.reset()

    def reset(self) -> None:
        self.noise_floor: float | None = None
        self._hold = 0
        self._gain = 1.0

    def process(self, data: np.ndarray) -> np.ndarray:
        if data.size == 0:
            return data
        energy = float(np.mean(np.square(data)))
        if self.noise_floor is None:
            self.noise_floor = energy
        coeff = self.floor_fall if energy < self.noise_floor else self.floor_rise
        self.noise_floor += (energy - self.noise_floor) * coeff

        if energy >= max(self.noise_floor, 1e-10) * self.open_ratio:
            self._hold = self.hold_chunks
        elif self._hold > 0:
            self._hold -= 1
        target = 1.0 if self._hold > 0 else self.attenuation
        prev = self._gain
        self._gain = target
        return _apply_gain_ramp(data, prev, target)


def process_chunk_audio(
    data: np.ndarray,
    sample_rate: int,
    *,
    mono_mode: str | None = None,
    denoise: bool | None = None,
    normalize: bool | None = None,
    gate: NoiseGate | None = None,
    agc: AutoGain | None = None,
) -> tuple[np.ndarray, int]:
    """Full DSP chain → mono float32 @ 16 kHz.

    Pass stateful ``gate``/``agc`` when processing a live stream of short chunks;
    without them the stateless whole-clip helpers are used.
    """
    mono_mode = mono_mode or getattr(config, "DSP_MONO_MODE", "energy")
    do_denoise = config.DSP_DENOISE_ENABLED if denoise is None else denoise
    do_norm = config.DSP_NORMALIZE_ENABLED if normalize is None else normalize

    mono = to_mono(data, mode=mono_mode)
    resampled = resample(mono, sample_rate, _TARGET_SR)
    if do_denoise:
        if gate is not None:
            resampled = gate.process(resampled)
        else:
            resampled = light_denoise(resampled, _TARGET_SR)
    if do_norm:
        if agc is not None:
            resampled = agc.process(resampled)
        else:
            resampled = normalize_peak(
                resampled, peak=getattr(config, "DSP_TARGET_PEAK", 0.9)
            )
    return resampled, _TARGET_SR


def float32_to_pcm16(data: np.ndarray) -> bytes:
    clipped = np.clip(data, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()
