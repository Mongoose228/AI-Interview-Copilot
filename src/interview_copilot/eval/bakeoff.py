"""Compare multiple STT backends on the golden set (bake-off)."""

from __future__ import annotations

import json
from pathlib import Path

from ..logging_config import logger
from ..stt.deepgram import DeepgramEngine
from ..stt.groq_engine import GroqWhisperEngine
from ..stt.soniox import SonioxEngine
from ..stt.whisper_engine import WhisperEngine
from .harness import DEFAULT_GOLDEN, run_harness


def _candidates() -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = [("whisper", WhisperEngine())]
    dg = DeepgramEngine()
    if dg.is_available:
        out.append(("deepgram", dg))
    else:
        logger.info("Bake-off: skip deepgram (no API key)")
    groq = GroqWhisperEngine()
    if groq.is_available:
        out.append(("groq", groq))
    else:
        logger.info("Bake-off: skip groq (no API key)")
    soniox = SonioxEngine()
    if soniox.is_available:
        out.append(("soniox", soniox))
    else:
        logger.info("Bake-off: skip soniox (no API key)")
    return out


def run_bakeoff(golden_dir: Path | None = None) -> dict:
    golden_dir = golden_dir or DEFAULT_GOLDEN
    reports = {}
    for name, engine in _candidates():
        logger.info(f"Bake-off running backend={name}")
        reports[name] = run_harness(engine, golden_dir=golden_dir)

    ranking = sorted(
        (
            {
                "backend": name,
                "mean_wer": data["mean_wer"],
                "mean_latency_s": data["mean_latency_s"],
                "scored_clip_count": data["scored_clip_count"],
            }
            for name, data in reports.items()
        ),
        key=lambda r: (r["mean_wer"], r["mean_latency_s"]),
    )
    return {"golden_dir": str(golden_dir), "ranking": ranking, "reports": reports}


def write_bakeoff(path: Path, golden_dir: Path | None = None) -> dict:
    result = run_bakeoff(golden_dir)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
