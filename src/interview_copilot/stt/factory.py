"""Build STT engines from config."""

from __future__ import annotations

from ..config import config
from ..logging_config import logger
from .base import SttEngine
from .deepgram import DeepgramEngine
from .dual_path import DualPathSttEngine
from .failover import FailoverSttEngine
from .groq_engine import GroqWhisperEngine
from .soniox import SonioxEngine
from .whisper_engine import WhisperEngine


def _build_primary(name: str) -> SttEngine:
    name = (name or "whisper").lower().strip()
    if name == "deepgram":
        eng = DeepgramEngine()
        if not eng.is_available:
            logger.warning("STT_BACKEND=deepgram but DEEPGRAM_API_KEY is empty")
        return eng
    if name == "groq":
        eng = GroqWhisperEngine()
        if not eng.is_available:
            logger.warning("STT_BACKEND=groq but GROQ_API_KEY is empty")
        return eng
    if name == "soniox":
        eng = SonioxEngine()
        if not eng.is_available:
            logger.warning("STT_BACKEND=soniox but SONIOX_API_KEY is empty")
        return eng
    return WhisperEngine()


def _build_fallback(name: str) -> SttEngine:
    name = (name or "whisper").lower().strip()
    if name == "deepgram":
        return DeepgramEngine()
    if name == "groq":
        return GroqWhisperEngine()
    if name == "soniox":
        return SonioxEngine()
    return WhisperEngine(model_size=config.WHISPER_MODEL)


def _wrap_dual_path(engine: SttEngine, primary_name: str) -> SttEngine:
    if not config.STT_DUAL_PATH_ENABLED:
        return engine

    provisional_size = config.STT_PROVISIONAL_MODEL
    # Avoid loading the same model twice when already on tiny.
    if primary_name == "whisper" and config.WHISPER_MODEL == provisional_size:
        logger.info("Dual-path skipped: WHISPER_MODEL already equals provisional model")
        return engine

    try:
        local = WhisperEngine(
            model_size=provisional_size,
            skip_if_unavailable=True,
        )
        if not local.is_available:
            logger.warning(f"Dual-path provisional model '{provisional_size}' unavailable")
            return engine
        wrapped = DualPathSttEngine(final_engine=engine, provisional=local)
        logger.info(
            f"Dual-path STT enabled "
            f"(final={primary_name}/{getattr(config, 'WHISPER_MODEL', '?')}, "
            f"provisional={provisional_size})"
        )
        return wrapped
    except Exception as e:
        logger.warning(f"Dual-path local STT unavailable: {e}")
        return engine


def build_stt_engine() -> SttEngine:
    primary = _build_primary(config.STT_BACKEND)
    fallback_name = (config.STT_FALLBACK or "whisper").lower().strip()
    primary_name = (config.STT_BACKEND or "whisper").lower().strip()

    if fallback_name and fallback_name not in {"none", primary_name}:
        fallback = _build_fallback(fallback_name)
        engine: SttEngine = FailoverSttEngine(primary, fallback)
    else:
        engine = primary

    # Dual-path for local Whisper and for cloud primaries (tiny UI + quality final).
    if primary_name in {"whisper", "deepgram", "groq", "soniox"}:
        engine = _wrap_dual_path(engine, primary_name)

    logger.info(
        f"[STT] backend={config.STT_BACKEND} fallback={config.STT_FALLBACK} "
        f"streaming={engine.supports_streaming()} available={engine.is_available} "
        f"dual_path={isinstance(engine, DualPathSttEngine)}"
    )
    return engine
