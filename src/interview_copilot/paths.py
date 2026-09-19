"""Application data paths (APPDATA on Windows, XDG-style elsewhere)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def get_data_dir() -> Path:
    """Return the persistent application data directory, creating it if needed."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    data_dir = base / "interview_copilot"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_context_dir() -> Path:
    path = get_data_dir() / "context"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    path = get_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir() -> Path:
    path = get_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_state_path(filename: str = "state.json") -> Path:
    return get_data_dir() / filename


def get_privacy_state_path() -> Path:
    return get_data_dir() / ".privacy_state.json"


def get_profile_state_path() -> Path:
    return get_data_dir() / ".copilot_state.json"


def get_window_state_path() -> Path:
    return get_data_dir() / "window_state.json"


def find_env_file() -> Path | None:
    """Locate .env: CWD, then data-dir, then project root (editable install)."""
    candidates = [
        Path.cwd() / ".env",
        get_data_dir() / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _example_profile_sources() -> list[Path]:
    sources: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        sources.append(Path(meipass) / "context" / "example_profile.md")
    sources.append(
        Path(__file__).resolve().parent.parent.parent / "context" / "example_profile.md"
    )
    return sources


def ensure_example_profile(context_dir: Path | None = None) -> None:
    """Copy bundled example_profile.md into context dir if missing."""
    ctx = context_dir or get_context_dir()
    dest = ctx / "example_profile.md"
    if dest.exists():
        return

    for src in _example_profile_sources():
        if src.is_file():
            shutil.copy2(src, dest)
            return

    dest.write_text(
        "# Example Profile\n\nReplace this with your real candidate profile.\n",
        encoding="utf-8",
    )
