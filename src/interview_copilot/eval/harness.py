"""Batch STT bench over testdata/stt_golden."""

from __future__ import annotations

import json
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..logging_config import logger
from ..models import SpeechPhrase
from ..stt.base import SttEngine
from .wer import cer, term_hits, wer

DEFAULT_GOLDEN = Path(__file__).resolve().parents[3] / "testdata" / "stt_golden"
DEFAULT_TECH_TERMS = [
    "Kubernetes",
    "PostgreSQL",
    "idempotency",
    "Redis",
    "caching",
]


@dataclass
class ClipResult:
    id: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    latency_s: float
    audio_duration_s: float
    term_hits: dict[str, bool]
    tags: list[str]


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def load_manifest(golden_dir: Path) -> dict:
    manifest_path = golden_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def run_harness(
    engine: SttEngine,
    golden_dir: Path | None = None,
    tech_terms: list[str] | None = None,
) -> dict:
    golden_dir = golden_dir or DEFAULT_GOLDEN
    tech_terms = tech_terms or DEFAULT_TECH_TERMS
    manifest = load_manifest(golden_dir)
    results: list[ClipResult] = []

    for clip in manifest.get("clips", []):
        audio_path = golden_dir / clip["audio"]
        if not audio_path.exists():
            logger.warning(f"[Harness] missing audio: {audio_path}")
            continue
        audio, sr = _load_wav(audio_path)
        duration = len(audio) / float(sr) if sr else 0.0
        now = time.time()
        phrase = SpeechPhrase(
            id=uuid.uuid4(),
            audio_data=audio.astype(np.float32),
            duration_s=duration,
            captured_at=now,
            vad_end_at=now,
            is_final=True,
            revision=1,
        )
        t0 = time.time()
        transcript = engine.transcribe(phrase)
        latency = time.time() - t0
        hyp = transcript.text_en or ""
        ref = clip.get("reference", "")
        results.append(
            ClipResult(
                id=clip["id"],
                reference=ref,
                hypothesis=hyp,
                wer=wer(ref, hyp),
                cer=cer(ref, hyp),
                latency_s=latency,
                audio_duration_s=duration,
                term_hits=term_hits(ref, hyp, tech_terms),
                tags=list(clip.get("tags") or []),
            )
        )

    scored = [r for r in results if r.reference.strip()]
    mean_wer = sum(r.wer for r in scored) / len(scored) if scored else 0.0
    mean_latency = sum(r.latency_s for r in results) / len(results) if results else 0.0
    term_misses: dict[str, int] = {}
    for r in results:
        for term, hit in r.term_hits.items():
            if not hit:
                term_misses[term] = term_misses.get(term, 0) + 1

    return {
        "golden_dir": str(golden_dir),
        "clip_count": len(results),
        "scored_clip_count": len(scored),
        "mean_wer": mean_wer,
        "mean_latency_s": mean_latency,
        "term_misses": term_misses,
        "clips": [asdict(r) for r in results],
    }
