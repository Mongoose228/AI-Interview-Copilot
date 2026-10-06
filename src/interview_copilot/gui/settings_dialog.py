"""Minimal in-app settings for STT / API keys."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..config import config
from ..settings_store import load_prefs, save_prefs


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)
        self.setStyleSheet("""
            QDialog { background-color: #2a2a2a; color: #E0E0E0; }
            QLabel { color: #CCCCCC; }
            QLineEdit, QComboBox {
                background-color: #1e1e1e; color: #E0E0E0;
                border: 1px solid #555; border-radius: 4px; padding: 4px;
            }
        """)

        prefs = load_prefs()
        secrets = prefs.get("secrets") or {}
        settings = prefs.get("settings") or {}

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.openrouter = QLineEdit(
            secrets.get("OPENROUTER_API_KEY") or config.OPENROUTER_API_KEY or ""
        )
        self.openrouter.setEchoMode(QLineEdit.Password)
        self.deepgram = QLineEdit(
            secrets.get("DEEPGRAM_API_KEY") or config.DEEPGRAM_API_KEY or ""
        )
        self.deepgram.setEchoMode(QLineEdit.Password)
        self.groq = QLineEdit(secrets.get("GROQ_API_KEY") or config.GROQ_API_KEY or "")
        self.groq.setEchoMode(QLineEdit.Password)
        self.deepl = QLineEdit(secrets.get("DEEPL_API_KEY") or config.DEEPL_API_KEY or "")
        self.deepl.setEchoMode(QLineEdit.Password)

        self.stt_backend = QComboBox()
        # Local first; cloud backends optional / paid — not recommended by default.
        self.stt_backend.addItems(["whisper", "deepgram", "groq", "soniox"])
        current = settings.get("STT_BACKEND") or config.STT_BACKEND or "whisper"
        idx = self.stt_backend.findText(current)
        if idx >= 0:
            self.stt_backend.setCurrentIndex(idx)

        self.mic_device = QLineEdit(
            settings.get("AUDIO_MIC_DEVICE") or config.AUDIO_MIC_DEVICE or ""
        )
        self.mic_device.setPlaceholderText("Mic device id (optional, dual-channel)")

        self.dual_path = QCheckBox(
            "Fast provisional STT (tiny.en) + quality final (base.en / selected backend)"
        )
        self.dual_path.setChecked(
            bool(settings.get("STT_DUAL_PATH_ENABLED", config.STT_DUAL_PATH_ENABLED))
        )
        self.dual_channel = QCheckBox(
            "Also capture microphone (candidate) — skips LLM for mic speech"
        )
        self.dual_channel.setChecked(
            bool(settings.get("AUDIO_DUAL_CHANNEL", config.AUDIO_DUAL_CHANNEL))
        )
        self.denoise = QCheckBox("Light denoise")
        self.denoise.setChecked(
            bool(settings.get("DSP_DENOISE_ENABLED", config.DSP_DENOISE_ENABLED))
        )

        form.addRow("OpenRouter API key", self.openrouter)
        form.addRow("Deepgram API key (optional, paid)", self.deepgram)
        form.addRow("Groq API key (optional)", self.groq)
        form.addRow("DeepL API key", self.deepl)
        form.addRow("STT backend", self.stt_backend)
        form.addRow("Mic device id", self.mic_device)
        layout.addLayout(form)
        layout.addWidget(self.dual_path)
        layout.addWidget(self.dual_channel)
        layout.addWidget(self.denoise)
        layout.addWidget(
            QLabel(
                "Recommended: STT = whisper (local, free). "
                "Deepgram/Groq/Soniox are optional paid cloud backends. "
                "Backend changes apply on next app restart."
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(buttons)
        layout.addLayout(row)

    def _save(self) -> None:
        prefs = load_prefs()
        prefs["secrets"] = {
            "OPENROUTER_API_KEY": self.openrouter.text().strip(),
            "DEEPGRAM_API_KEY": self.deepgram.text().strip(),
            "GROQ_API_KEY": self.groq.text().strip(),
            "DEEPL_API_KEY": self.deepl.text().strip(),
        }
        prefs["settings"] = {
            **(prefs.get("settings") or {}),
            "STT_BACKEND": self.stt_backend.currentText(),
            "STT_DUAL_PATH_ENABLED": self.dual_path.isChecked(),
            "AUDIO_DUAL_CHANNEL": self.dual_channel.isChecked(),
            "AUDIO_MIC_DEVICE": self.mic_device.text().strip() or None,
            "DSP_DENOISE_ENABLED": self.denoise.isChecked(),
        }
        save_prefs(prefs)
        # Hot-apply keys / flags that are safe without rebuild.
        for key, value in prefs["secrets"].items():
            if value:
                object.__setattr__(config, key, value)
        object.__setattr__(config, "STT_BACKEND", prefs["settings"]["STT_BACKEND"])
        object.__setattr__(
            config, "STT_DUAL_PATH_ENABLED", prefs["settings"]["STT_DUAL_PATH_ENABLED"]
        )
        object.__setattr__(
            config, "AUDIO_DUAL_CHANNEL", prefs["settings"]["AUDIO_DUAL_CHANNEL"]
        )
        object.__setattr__(
            config, "AUDIO_MIC_DEVICE", prefs["settings"]["AUDIO_MIC_DEVICE"]
        )
        object.__setattr__(
            config, "DSP_DENOISE_ENABLED", prefs["settings"]["DSP_DENOISE_ENABLED"]
        )
        self.accept()
