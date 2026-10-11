from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import ensure_example_profile, find_env_file, get_context_dir, get_data_dir


def _default_context_dir() -> str:
    ensure_example_profile()
    return str(get_context_dir())


_env_file = find_env_file()


class Settings(BaseSettings):
    # API Keys
    DEEPL_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "google/gemini-2.5-flash"
    NETWORK_CONNECT_TIMEOUT: int = 5
    NETWORK_READ_TIMEOUT: int = 30
    MAX_LLM_TOKENS: int = 250

    # STT backend selection
    STT_BACKEND: Literal["whisper", "deepgram", "groq", "soniox"] = "whisper"
    STT_FALLBACK: Literal["whisper", "deepgram", "groq", "soniox", "none"] = "whisper"
    STT_KEYTERMS_FROM_PROFILE: bool = True
    STT_DUAL_PATH_ENABLED: bool = True
    STT_PROVISIONAL_MODEL: str = "tiny.en"
    # When using cloud streaming, prefer Deepgram utterance boundaries over Silero cuts.
    STT_CLOUD_ENDPOINTING: bool = True

    DEEPGRAM_API_KEY: str | None = None
    DEEPGRAM_MODEL: str = "nova-3"
    DEEPGRAM_LANGUAGE: str = "en"
    DEEPGRAM_ENDPOINTING_MS: int = 300
    DEEPGRAM_UTTERANCE_END_MS: int = 1000

    GROQ_API_KEY: str | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"

    SONIOX_API_KEY: str | None = None
    SONIOX_MODEL: str = "stt-rt-preview"

    # Whisper Settings
    WHISPER_MODEL: str = "base.en"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_BEAM_SIZE: int = 1
    WHISPER_CPU_THREADS: int = 4

    # Audio Settings
    AUDIO_DEVICE: str | None = None
    AUDIO_MIC_DEVICE: str | None = None
    AUDIO_DUAL_CHANNEL: bool = False
    # When mic VAD detects the candidate speaking, skip loopback STT enqueue
    # so your own voice (echoed from speakers) does not become "interviewer" text.
    AUDIO_SUPPRESS_LOOPBACK_WHILE_MIC: bool = True
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_MS: int = 20

    # DSP
    DSP_ENABLED: bool = True
    DSP_MONO_MODE: Literal["energy", "mean", "left", "right"] = "energy"
    DSP_NORMALIZE_ENABLED: bool = True
    DSP_DENOISE_ENABLED: bool = True
    DSP_TARGET_PEAK: float = 0.9

    # VAD Settings
    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_MS: int = 250
    VAD_SILENCE_MS: int = 350
    VAD_SPEECH_PAD_MS: int = 150
    VAD_MAX_PHRASE_SECONDS: int = 12

    # Debounce: seconds of silence after last transcript before sending to LLM.
    # Lower default when using fast cloud STT; raise for Whisper-only CPU.
    LLM_DEBOUNCE_DELAY_S: float = 0.25
    # Upper bound on waiting for VAD speech-end before flushing to the LLM anyway
    # (continuous background audio can keep VAD "speaking" indefinitely).
    LLM_MAX_WAIT_FOR_SPEECH_END_S: float = 4.0

    # Partial STT settings (bounded to one in-flight CPU inference)
    PARTIAL_TRANSCRIPTION_ENABLED: bool = True
    PARTIAL_TRANSCRIPTION_INTERVAL_MS: int = 800
    PARTIAL_TRANSCRIPTION_MIN_AUDIO_MS: int = 700

    # Translation Settings
    TRANSLATION_BACKEND: Literal["none", "deepl", "nllb"] = "none"
    NLLB_MODEL: str = "facebook/nllb-200-distilled-600M"

    # Application Settings
    CONTEXT_DIR: str = ""
    LOG_OBFUSCATION_ENABLED: bool = True
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(_env_file) if _env_file else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context) -> None:
        if not self.CONTEXT_DIR:
            object.__setattr__(self, "CONTEXT_DIR", _default_context_dir())
        else:
            Path(self.CONTEXT_DIR).mkdir(parents=True, exist_ok=True)
            ensure_example_profile(Path(self.CONTEXT_DIR))
        get_data_dir()  # ensure exists


config = Settings()
