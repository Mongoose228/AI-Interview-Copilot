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


def process_chunk_audio(
    data: np.ndarray,
    sample_rate: int,
    *,
    mono_mode: str | None = None,
    denoise: bool | None = None,
    normalize: bool | None = None,
) -> tuple[np.ndarray, int]:
    """Full DSP chain → mono float32 @ 16 kHz."""
    mono_mode = mono_mode or getattr(config, "DSP_MONO_MODE", "energy")
    do_denoise = config.DSP_DENOISE_ENABLED if denoise is None else denoise
    do_norm = config.DSP_NORMALIZE_ENABLED if normalize is None else normalize

    mono = to_mono(data, mode=mono_mode)
    resampled = resample(mono, sample_rate, _TARGET_SR)
    if do_denoise:
        resampled = light_denoise(resampled, _TARGET_SR)
    if do_norm:
        resampled = normalize_peak(resampled, peak=getattr(config, "DSP_TARGET_PEAK", 0.9))
    return resampled, _TARGET_SR


def float32_to_pcm16(data: np.ndarray) -> bytes:
    clipped = np.clip(data, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()
