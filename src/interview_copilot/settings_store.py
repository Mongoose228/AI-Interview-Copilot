"""Persist API keys / preferences (keyring when available, else AppData JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .logging_config import logger
from .paths import get_data_dir

_SERVICE = "interview-copilot"
_KEYS = (
    "OPENROUTER_API_KEY",
    "DEEPGRAM_API_KEY",
    "GROQ_API_KEY",
    "SONIOX_API_KEY",
    "DEEPL_API_KEY",
)
_plaintext_warned = False


def _prefs_path() -> Path:
    return get_data_dir() / "user_prefs.json"


def _keyring_get(name: str) -> str | None:
    try:
        import keyring

        return keyring.get_password(_SERVICE, name)
    except Exception:
        return None


def _keyring_set(name: str, value: str | None) -> bool:
    try:
        import keyring

        if not value:
            try:
                keyring.delete_password(_SERVICE, name)
            except Exception:
                pass
            return True
        keyring.set_password(_SERVICE, name, value)
        return True
    except Exception as e:
        logger.debug(f"keyring unavailable: {e}")
        return False


def load_prefs() -> dict[str, Any]:
    path = _prefs_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load prefs: {e}")
    secrets = data.get("secrets") or {}
    for key in _KEYS:
        from_kr = _keyring_get(key)
        if from_kr:
            secrets[key] = from_kr
    data["secrets"] = secrets
    return data


def save_prefs(prefs: dict[str, Any]) -> None:
    global _plaintext_warned
    secrets = dict(prefs.get("secrets") or {})
    stored_in_keyring: list[str] = []
    for key in _KEYS:
        val = secrets.get(key)
        if _keyring_set(key, val if val else None):
            stored_in_keyring.append(key)
    # Do not write secrets that made it into keyring into plaintext JSON.
    plaintext_secrets = {
        k: v for k, v in secrets.items() if k not in stored_in_keyring and v
    }
    out = {**prefs, "secrets": plaintext_secrets}
    path = _prefs_path()
    if not _plaintext_warned and any(k in _KEYS for k in plaintext_secrets):
        _plaintext_warned = True
        logger.warning(
            f"System keyring unavailable; API keys are stored in plaintext at {path}"
        )
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save prefs: {e}")


def apply_prefs_to_config() -> None:
    """Overlay persisted prefs onto the live config object."""
    from . import config as config_mod

    cfg = config_mod.config
    fields = type(cfg).model_fields
    prefs = load_prefs()
    secrets = prefs.get("secrets") or {}
    for key, value in secrets.items():
        if value and isinstance(value, str) and key in fields:
            object.__setattr__(cfg, key, value)
    settings = prefs.get("settings") or {}
    for key, value in settings.items():
        field = fields.get(key)
        if field is None:
            continue
        try:
            TypeAdapter(field.annotation).validate_python(value, strict=True)
        except ValidationError:
            logger.warning(f"Ignoring saved setting {key}={value!r}: invalid type/value")
            continue
        object.__setattr__(cfg, key, value)
