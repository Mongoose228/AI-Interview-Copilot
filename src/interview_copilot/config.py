from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    DEEPL_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "google/gemini-2.5-flash"
    NETWORK_CONNECT_TIMEOUT: int = 5
    NETWORK_READ_TIMEOUT: int = 30

    # Whisper Settings
    WHISPER_MODEL: str = "small.en"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_BEAM_SIZE: int = 1

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
    VAD_MAX_PHRASE_SECONDS: int = 30

    # Translation Settings
    TRANSLATION_BACKEND: Literal["none", "deepl", "nllb"] = "none"
    NLLB_MODEL: str = "facebook/nllb-200-distilled-600M"

    # Application Settings
    CONTEXT_DIR: str = "./context"
    LOG_OBFUSCATION_ENABLED: bool = True
    TEXT_LOGGING_ENABLED: bool = False
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


config = Settings()
