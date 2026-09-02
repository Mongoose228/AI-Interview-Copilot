from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # API Keys
    DEEPL_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "google/gemini-2.5-flash"
    NETWORK_CONNECT_TIMEOUT: int = 5
    NETWORK_READ_TIMEOUT: int = 30
    MAX_LLM_TOKENS: int = 250

    # Whisper Settings
    WHISPER_MODEL: str = "small.en"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_BEAM_SIZE: int = 1
    WHISPER_CPU_THREADS: int = 4

    # Audio Settings
    AUDIO_DEVICE: str | None = None
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_MS: int = 30
    AUDIO_BACKEND: str = "soundcard"

    # VAD Settings
    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_MS: int = 250
    VAD_SILENCE_MS: int = 600
    VAD_SPEECH_PAD_MS: int = 150
    VAD_MAX_PHRASE_SECONDS: int = 12

    # Debounce: seconds of silence after last transcript before sending to LLM.
    # Higher values = better question merging, but higher latency.
    LLM_DEBOUNCE_DELAY_S: float = 1.2

    # Translation Settings
    TRANSLATION_BACKEND: Literal["none", "deepl", "nllb"] = "none"
    NLLB_MODEL: str = "facebook/nllb-200-distilled-600M"

    # Application Settings
    CONTEXT_DIR: str = str(_ROOT_DIR / "context")
    LOG_OBFUSCATION_ENABLED: bool = True
    TEXT_LOGGING_ENABLED: bool = False
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=str(_ROOT_DIR / ".env"), env_file_encoding="utf-8")


config = Settings()
