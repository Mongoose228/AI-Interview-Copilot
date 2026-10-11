import json

import pytest

from interview_copilot import settings_store
from interview_copilot.config import config


class _FakeLogger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg):
        self.warnings.append(msg)

    def debug(self, msg):
        pass

    def error(self, msg):
        pass


@pytest.fixture
def store(tmp_path, monkeypatch):
    prefs_file = tmp_path / "user_prefs.json"
    monkeypatch.setattr(settings_store, "_prefs_path", lambda: prefs_file)
    monkeypatch.setattr(settings_store, "_keyring_get", lambda name: None)
    monkeypatch.setattr(settings_store, "_keyring_set", lambda name, value: False)
    monkeypatch.setattr(settings_store, "_plaintext_warned", False)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(settings_store, "logger", fake_logger)
    return prefs_file, fake_logger


def _preserve_config(monkeypatch, *keys):
    for key in keys:
        monkeypatch.setattr(config, key, getattr(config, key))


def test_soniox_key_is_persisted():
    assert "SONIOX_API_KEY" in settings_store._KEYS


def test_apply_prefs_skips_invalid_types(store, monkeypatch):
    prefs_file, fake_logger = store
    _preserve_config(
        monkeypatch,
        "STT_DUAL_PATH_ENABLED",
        "STT_BACKEND",
        "LLM_DEBOUNCE_DELAY_S",
        "AUDIO_MIC_DEVICE",
        "DSP_DENOISE_ENABLED",
        "SONIOX_API_KEY",
    )
    monkeypatch.setattr(config, "STT_DUAL_PATH_ENABLED", True)
    monkeypatch.setattr(config, "STT_BACKEND", "whisper")
    monkeypatch.setattr(config, "AUDIO_MIC_DEVICE", "mic-1")
    prefs_file.write_text(
        json.dumps(
            {
                "secrets": {"SONIOX_API_KEY": "sx-key"},
                "settings": {
                    "STT_DUAL_PATH_ENABLED": "yes",
                    "STT_BACKEND": "bogus",
                    "LLM_DEBOUNCE_DELAY_S": 1,
                    "AUDIO_MIC_DEVICE": None,
                    "DSP_DENOISE_ENABLED": False,
                    "NOT_A_SETTING": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    settings_store.apply_prefs_to_config()

    assert config.STT_DUAL_PATH_ENABLED is True
    assert config.STT_BACKEND == "whisper"
    assert config.LLM_DEBOUNCE_DELAY_S == 1
    assert config.AUDIO_MIC_DEVICE is None
    assert config.DSP_DENOISE_ENABLED is False
    assert config.SONIOX_API_KEY == "sx-key"
    assert not hasattr(config, "NOT_A_SETTING")
    assert len(fake_logger.warnings) == 2


def test_plaintext_fallback_warns_once(store):
    prefs_file, fake_logger = store
    settings_store.save_prefs({"secrets": {"GROQ_API_KEY": "g"}})
    settings_store.save_prefs({"secrets": {"GROQ_API_KEY": "g2"}})

    assert len([w for w in fake_logger.warnings if "plaintext" in w]) == 1
    data = json.loads(prefs_file.read_text(encoding="utf-8"))
    assert data["secrets"] == {"GROQ_API_KEY": "g2"}


def test_keyring_secrets_not_written_to_json(store, monkeypatch):
    prefs_file, fake_logger = store
    monkeypatch.setattr(settings_store, "_keyring_set", lambda name, value: True)
    settings_store.save_prefs({"secrets": {"GROQ_API_KEY": "g"}, "settings": {}})

    data = json.loads(prefs_file.read_text(encoding="utf-8"))
    assert data["secrets"] == {}
    assert fake_logger.warnings == []
