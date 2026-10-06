#!/usr/bin/env python3
"""Generate golden WAVs for STT harness (SAPI TTS on Windows, tone fallback elsewhere)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "testdata" / "stt_golden"
SR = 16000

CLIPS = [
    {
        "id": "silence_01",
        "audio": "silence_01.wav",
        "reference": "",
        "tags": ["silence"],
        "tts": None,
    },
    {
        "id": "tech_terms_01",
        "audio": "tech_terms_01.wav",
        "reference": "Explain Kubernetes PostgreSQL idempotency and Redis caching.",
        "tags": ["tech", "short", "tts"],
        "tts": "Explain Kubernetes, PostgreSQL, idempotency, and Redis caching.",
    },
    {
        "id": "tech_terms_02",
        "audio": "tech_terms_02.wav",
        "reference": "How do you design a rate limiter with Redis and idempotent APIs?",
        "tags": ["tech", "short", "tts"],
        "tts": "How do you design a rate limiter with Redis and idempotent A P Is?",
    },
    {
        "id": "long_q_01",
        "audio": "long_q_01.wav",
        "reference": (
            "Walk me through debugging a production outage involving "
            "Kubernetes pods and a PostgreSQL connection pool."
        ),
        "tags": ["tech", "long", "tts"],
        "tts": (
            "Walk me through debugging a production outage involving "
            "Kubernetes pods and a PostgreSQL connection pool."
        ),
    },
    {
        "id": "filler_01",
        "audio": "filler_01.wav",
        "reference": "um",
        "tags": ["filler", "tts"],
        "tts": "um",
    },
]


def write_wav(path: Path, samples: np.ndarray) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def tone_burst(duration_s: float, freq: float = 440.0, amp: float = 0.2) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    envelope = np.linspace(0, 1, len(t))
    envelope = np.minimum(envelope, envelope[::-1])
    envelope = np.clip(envelope * 8, 0, 1)
    return (amp * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _load_wav_any(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported width {width}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def _resample(data: np.ndarray, src_sr: int, dst_sr: int = SR) -> np.ndarray:
    if src_sr == dst_sr:
        return data.astype(np.float32)
    try:
        import soxr

        return soxr.resample(data.astype(np.float32), src_sr, dst_sr).astype(np.float32)
    except Exception:
        duration = len(data) / float(src_sr)
        n_out = max(1, int(round(duration * dst_sr)))
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, data).astype(np.float32)


def sapi_speak_to_wav(text: str, out_path: Path) -> bool:
    """Use Windows System.Speech to synthesize WAV, then resample to 16 kHz mono."""
    if sys.platform != "win32":
        return False
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "sapi.wav"
        # Escape for PowerShell single-quoted string
        safe = text.replace("'", "''")
        ps = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = -1
$synth.SetOutputToWaveFile('{raw_path}')
$synth.Speak('{safe}')
$synth.Dispose()
"""
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
        if not raw_path.exists():
            return False
        data, sr = _load_wav_any(raw_path)
        write_wav(out_path, _resample(data, sr, SR))
        return True


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    used_tts = 0
    for clip in CLIPS:
        path = ROOT / clip["audio"]
        if not clip["tts"]:
            write_wav(path, np.zeros(SR, dtype=np.float32))
            continue
        if sapi_speak_to_wav(clip["tts"], path):
            used_tts += 1
        else:
            # Fallback tone — harness I/O only, WER will be ~1.0
            write_wav(path, tone_burst(2.0 if "long" in clip["tags"] else 1.2))

    manifest = {
        "version": 1,
        "sample_rate": SR,
        "notes": (
            "Prefer Windows SAPI TTS clips for local WER. "
            "Replace with real Zoom/Meet loopback for production validation. "
            "Do not commit PII."
        ),
        "clips": [
            {k: v for k, v in c.items() if k != "tts"} for c in CLIPS
        ],
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(CLIPS)} clips to {ROOT} (tts={used_tts})")


if __name__ == "__main__":
    main()
