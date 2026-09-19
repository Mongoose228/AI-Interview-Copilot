"""Privacy notice banner shown on first launch."""

import json

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..logging_config import logger
from ..paths import get_privacy_state_path

_STATE_KEY = "privacy_accepted"


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

        self.setStyleSheet("""
            PrivacyBanner {
                background-color: rgba(50, 40, 20, 0.8);
                border: 1px solid rgba(255, 183, 77, 0.5);
                border-radius: 8px;
                margin-bottom: 10px;
            }
            QLabel { background: transparent; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        lbl = QLabel(
            "⚠️ <b>Privacy Notice</b><br>"
            "This app sends interview transcripts and your candidate profile "
            "to <b>OpenRouter API</b> (cloud) to generate AI suggestions. "
            "No data is stored on the server beyond the API call duration."
        )
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #FFB74D; font-size: 12px;")
        layout.addWidget(lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn = QPushButton("I Understand")
        btn.setFixedWidth(140)
        btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 183, 77, 0.3);
                color: #FFB74D;
                border: 1px solid rgba(255, 183, 77, 0.5);
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 183, 77, 0.5);
                color: #FFFFFF;
            }
        """)
        btn.clicked.connect(self._accept)
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)

    def _accept(self):
        _save_accepted()
        self.setVisible(False)
