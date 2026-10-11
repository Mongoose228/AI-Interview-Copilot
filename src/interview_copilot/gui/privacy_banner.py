"""Privacy notice banner shown on first launch."""

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..logging_config import logger
from ..paths import get_privacy_state_path
from . import theme

_STATE_KEY = "privacy_accepted_v3"


def _is_accepted() -> bool:
    path = get_privacy_state_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
                return state.get(_STATE_KEY, False)
        except Exception:
            pass
    return False


def _save_accepted():
    path = get_privacy_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({_STATE_KEY: True}, f)
    except Exception as e:
        logger.warning(f"Failed to save privacy state: {e}")


class PrivacyBanner(QFrame):
    """One-time privacy notice shown before first use."""

    def __init__(self, parent=None):
        super().__init__(parent)

        if _is_accepted():
            self.setVisible(False)
            return

        self.setStyleSheet(theme.privacy_banner_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel("Privacy notice")
        title.setStyleSheet(
            f"color: {theme.TEXT_WARN}; font-size: 13px; font-weight: 600;"
        )
        layout.addWidget(title)

        lbl = QLabel(
            "By default, speech is transcribed <b>locally with Whisper</b> "
            "(audio stays on your machine). "
            "Suggestions send transcripts and your candidate profile to "
            "<b>OpenRouter</b>. Optional paid STT (Deepgram / Groq / Soniox) "
            "or DeepL translation only run if you enable them and add keys."
        )
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn = QPushButton("I Understand")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedWidth(140)
        btn.setStyleSheet(theme.privacy_button_style())
        btn.clicked.connect(self._accept)
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)

    def _accept(self):
        _save_accepted()
        self.setVisible(False)
