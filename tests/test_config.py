from interview_copilot.config import Settings


def test_default_config(monkeypatch):
    # clear env to test defaults
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("TRANSLATION_BACKEND", raising=False)

    # _env_file=None prevents reading .env from disk
    config = Settings(_env_file=None)
    assert config.DEEPL_API_KEY is None
    assert config.OPENROUTER_API_KEY is None
    assert config.OPENROUTER_MODEL == "google/gemini-2.5-flash"
    assert config.WHISPER_MODEL == "base.en"
    assert config.NETWORK_CONNECT_TIMEOUT == 5
    assert config.NETWORK_READ_TIMEOUT == 30
    assert config.TRANSLATION_BACKEND == "none"
    assert config.VAD_THRESHOLD == 0.5
    assert config.VAD_SILENCE_MS == 350
    assert config.WHISPER_BEAM_SIZE == 1
    assert config.WHISPER_CPU_THREADS == 4
    assert config.LLM_DEBOUNCE_DELAY_S == 0.25
    assert config.AUDIO_CHUNK_MS == 20
    assert config.STT_BACKEND == "whisper"
    assert config.STT_DUAL_PATH_ENABLED is True
    assert config.STT_PROVISIONAL_MODEL == "tiny.en"
    assert config.AUDIO_SUPPRESS_LOOPBACK_WHILE_MIC is True
    assert config.PARTIAL_TRANSCRIPTION_ENABLED is True
    assert config.PARTIAL_TRANSCRIPTION_INTERVAL_MS == 800
    assert config.PARTIAL_TRANSCRIPTION_MIN_AUDIO_MS == 700
    assert config.DSP_ENABLED is True


def test_custom_config(monkeypatch):
    monkeypatch.setenv("DEEPL_API_KEY", "test_key")
    monkeypatch.setenv("WHISPER_MODEL", "distil-large-v3")
    monkeypatch.setenv("NETWORK_CONNECT_TIMEOUT", "3")
    monkeypatch.setenv("NETWORK_READ_TIMEOUT", "20")
    monkeypatch.setenv("TRANSLATION_BACKEND", "deepl")
    monkeypatch.setenv("VAD_THRESHOLD", "0.7")

    config = Settings(_env_file=None)
    assert config.DEEPL_API_KEY == "test_key"
    assert config.WHISPER_MODEL == "distil-large-v3"
    assert config.NETWORK_CONNECT_TIMEOUT == 3
    assert config.NETWORK_READ_TIMEOUT == 20
    assert config.TRANSLATION_BACKEND == "deepl"
    assert config.VAD_THRESHOLD == 0.7
