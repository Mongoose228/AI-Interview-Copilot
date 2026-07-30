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
    assert config.WHISPER_MODEL == "small.en"
    assert config.NETWORK_CONNECT_TIMEOUT == 5
    assert config.NETWORK_READ_TIMEOUT == 30
    assert config.TRANSLATION_BACKEND == "none"
    assert config.VAD_THRESHOLD == 0.5
    assert config.VAD_SILENCE_MS == 600
    assert config.WHISPER_BEAM_SIZE == 1


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
