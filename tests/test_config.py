from interview_copilot.config import Settings


def test_default_config(monkeypatch):
    # clear env to test defaults
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    config = Settings()
    assert config.DEEPL_API_KEY is None
    assert config.OPENROUTER_API_KEY is None
    assert config.WHISPER_MODEL == "small.en"
    assert config.NETWORK_TIMEOUT_SECONDS == 5


def test_custom_config(monkeypatch):
    monkeypatch.setenv("DEEPL_API_KEY", "test_key")
    monkeypatch.setenv("WHISPER_MODEL", "distil-large-v3")
    monkeypatch.setenv("NETWORK_TIMEOUT_SECONDS", "10")

    config = Settings()
    assert config.DEEPL_API_KEY == "test_key"
    assert config.WHISPER_MODEL == "distil-large-v3"
    assert config.NETWORK_TIMEOUT_SECONDS == 10
