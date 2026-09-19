import ctypes
import html
import json
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

from ..models import PipelineResult, SuggestionStatus
from ..paths import get_window_state_path
from .privacy_banner import PrivacyBanner

_MAX_RESULT_CARDS = 50
_TRANSIENT_MS = 7000
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def _esc(text: str) -> str:
    return html.escape(text or "", quote=True)


class ResultWidget(QFrame):
    """A widget to display a single pipeline result."""

    def __init__(self, result: PipelineResult, parent=None):
        super().__init__(parent)
        self.phrase_id = result.id
        self.setStyleSheet("""
            ResultWidget {
                background-color: rgba(40, 40, 40, 0.4);
                border-radius: 10px;
                border: 1px solid rgba(80, 80, 80, 0.3);
                margin-bottom: 10px;
            }
            QLabel {
                background: transparent;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        lbl_en = QLabel(f"🎙️ <b>EN:</b> {_esc(result.transcript)}")
        lbl_en.setWordWrap(True)
        lbl_en.setStyleSheet("color: #CCCCCC; font-size: 13px;")
        layout.addWidget(lbl_en)

        self.lbl_ru = QLabel("")
        self.lbl_ru.setWordWrap(True)
        self.lbl_ru.setStyleSheet("color: #90CAF9; font-size: 13px;")
        self.lbl_ru.setVisible(False)
        layout.addWidget(self.lbl_ru)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line.setStyleSheet("background-color: rgba(100, 100, 100, 0.5);")
        layout.addWidget(self.line)

        self.lbl_ai = QLabel("🤔 <i>Thinking...</i>")
        self.lbl_ai.setWordWrap(True)
        self.lbl_ai.setStyleSheet("color: #888888; font-size: 13px; font-style: italic;")
        layout.addWidget(self.lbl_ai)

        self.lbl_ai_ru = QLabel("")
        self.lbl_ai_ru.setWordWrap(True)
        self.lbl_ai_ru.setStyleSheet("color: #81C784; font-size: 14px;")
        self.lbl_ai_ru.setVisible(False)
        layout.addWidget(self.lbl_ai_ru)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setFixedWidth(60)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.08);
                color: #AAAAAA;
                border: none;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
            }
            QPushButton:hover { color: #FFFFFF; background-color: rgba(255,255,255,0.15); }
        """)
        self.copy_btn.clicked.connect(self._copy_suggestion)
        btn_row.addWidget(self.copy_btn)
        layout.addLayout(btn_row)

        self.current_suggestion_text = ""

    def _copy_suggestion(self):
        if self.current_suggestion_text:
            QApplication.clipboard().setText(self.current_suggestion_text)

    def append_token(self, token: str):
        if not self.current_suggestion_text:
            self.lbl_ai.setStyleSheet("color: #4CAF50; font-size: 15px; font-weight: bold;")
        self.current_suggestion_text += token
        formatted_text = _esc(self.current_suggestion_text).replace("\n", "<br>")
        self.lbl_ai.setText(f"💡 <b>AI:</b> {formatted_text}")

    def update_suggestion(self, result: PipelineResult):
        status = getattr(result, "suggestion_status", SuggestionStatus.PENDING)
        suggestion = result.suggestion
        is_cancelled = getattr(result, "is_cancelled", False)

        if status == SuggestionStatus.INTERRUPTED or (
            is_cancelled and self.current_suggestion_text
        ):
            formatted = _esc(self.current_suggestion_text).replace("\n", "<br>")
            suffix = " <i>(interrupted)</i>" if self.current_suggestion_text else ""
            self.lbl_ai.setText(f"💡 <b>AI:</b> {formatted}{suffix}")
            self.lbl_ai.setStyleSheet("color: #FFB74D; font-size: 14px;")
            return

        if status == SuggestionStatus.CANCELLED or is_cancelled:
            self.lbl_ai.setText("⏸️ <i>Skipped (new question)</i>")
            self.lbl_ai.setStyleSheet(
                "color: #888888; font-size: 13px; font-style: italic;"
                " border: 1px solid rgba(136, 136, 136, 0.3);"
                " padding: 4px; border-radius: 4px;"
            )
            return

        if status == SuggestionStatus.SKIPPED_NO_KEY:
            self.lbl_ai.setText("🔑 <i>Suggestions disabled — set OPENROUTER_API_KEY</i>")
            self.lbl_ai.setStyleSheet("color: #FFB74D; font-size: 13px; font-style: italic;")
            return

        if status == SuggestionStatus.SKIPPED_NO_PROFILE:
            self.lbl_ai.setText("📄 <i>Suggestions disabled — create a candidate profile</i>")
            self.lbl_ai.setStyleSheet("color: #FFB74D; font-size: 13px; font-style: italic;")
            return

        if not suggestion or status == SuggestionStatus.FAILED:
            self.lbl_ai.setText("❌ <i>Suggestion failed.</i>")
            self.lbl_ai.setStyleSheet(
                "color: #F44336; font-size: 13px; font-style: italic;"
                " border: 1px solid rgba(244, 67, 54, 0.5);"
                " padding: 4px; border-radius: 4px;"
            )
            return

        text = suggestion.answer_en
        icon = " ⚠️" if suggestion.has_hedging else ""
        self.current_suggestion_text = text
        formatted_text = _esc(text).replace("\n", "<br>")
        self.lbl_ai.setStyleSheet("color: #4CAF50; font-size: 15px; font-weight: bold;")
        self.lbl_ai.setText(f"💡 <b>AI{icon}:</b> {formatted_text}")

    def set_translation(self, translation_ru: str):
        if translation_ru:
            self.lbl_ru.setText(f"🇷🇺 <b>RU:</b> {_esc(translation_ru)}")
            self.lbl_ru.setVisible(True)

    def set_answer_ru(self, answer_ru: str):
        if answer_ru:
            formatted = _esc(answer_ru).replace("\n", "<br>")
            self.lbl_ai_ru.setText(f"🇷🇺 <b>AI RU:</b> {formatted}")
            self.lbl_ai_ru.setVisible(True)


class CopilotMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Interview Copilot")
        self.resize(450, 600)
        self.setMinimumSize(320, 280)

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("""
            QWidget#MainContainer {
                background-color: rgba(25, 25, 25, 0.3);
                border-radius: 15px;
                border: 1px solid rgba(60, 60, 60, 0.4);
            }
        """)
        self.central_widget.setObjectName("MainContainer")
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)

        # Header
        self.header_layout = QHBoxLayout()
        self.title_lbl = QLabel("AI Interview Copilot")
        self.title_lbl.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        self.header_layout.addWidget(self.title_lbl)
        self.header_layout.addStretch()

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #888888; font-size: 12px;")
        self.header_layout.addWidget(self.status_lbl)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setFixedHeight(28)
        self.pause_btn.setStyleSheet(self._btn_style())
        self.header_layout.addWidget(self.pause_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedHeight(28)
        self.clear_btn.setStyleSheet(self._btn_style())
        self.header_layout.addWidget(self.clear_btn)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                color: #AAAAAA;
                border: none;
                border-radius: 14px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 60, 60, 0.6);
                color: #FFFFFF;
            }
        """)
        self.close_btn.clicked.connect(QApplication.instance().quit)
        self.header_layout.addWidget(self.close_btn)
        self.main_layout.addLayout(self.header_layout)

        # Controls row: profile + device
        controls = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(120)
        self.profile_combo.setStyleSheet(self._combo_style())
        controls.addWidget(QLabel("Profile:"))
        controls.addWidget(self.profile_combo, 1)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(140)
        self.device_combo.setStyleSheet(self._combo_style())
        controls.addWidget(QLabel("Device:"))
        controls.addWidget(self.device_combo, 1)
        self.main_layout.addLayout(controls)

        self.privacy_banner = PrivacyBanner()
        self.main_layout.addWidget(self.privacy_banner)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setContentsMargins(0, 0, 10, 0)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

        grip_row = QHBoxLayout()
        grip_row.addStretch()
        self.size_grip = QSizeGrip(self.central_widget)
        grip_row.addWidget(self.size_grip, 0, Qt.AlignBottom | Qt.AlignRight)
        self.main_layout.addLayout(grip_row)

        self._drag_pos = None
        self._result_widgets: list[ResultWidget] = []
        self._widget_map = {}
        self._has_persistent_error = False
        self._paused = False
        self._transient_timer = QTimer(self)
        self._transient_timer.setSingleShot(True)
        self._transient_timer.timeout.connect(self._clear_transient_status)

        shortcut_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut_esc.activated.connect(QApplication.instance().quit)

        shortcut_hide = QShortcut(QKeySequence("Ctrl+Shift+H"), self)
        shortcut_hide.activated.connect(self.toggle_visibility)

        self.title_lbl.installEventFilter(self)
        self.status_lbl.installEventFilter(self)

        self.clear_btn.clicked.connect(self.clear_cards)
        self.pause_btn.clicked.connect(self._on_pause_clicked)

        # Callbacks wired by app.py
        self.on_profile_changed = None
        self.on_device_changed = None
        self.on_pause_toggled = None
        self.on_clear_history = None

        self._restore_geometry()

    @staticmethod
    def _btn_style() -> str:
        return """
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                color: #CCCCCC;
                border: none;
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); color: #FFFFFF; }
        """

    @staticmethod
    def _combo_style() -> str:
        return """
            QComboBox {
                background-color: rgba(40,40,40,0.7);
                color: #DDDDDD;
                border: 1px solid rgba(80,80,80,0.5);
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2A2A;
                color: #EEEEEE;
                selection-background-color: #444444;
            }
        """

    def show(self):
        super().show()
        self._apply_display_affinity()

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def _apply_display_affinity(self):
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            result = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            from ..logging_config import logger
            if result:
                logger.info("Window display affinity set: excluded from screen capture.")
            else:
                logger.warning(
                    "Failed to set window display affinity. "
                    "Window may be visible in screen capture."
                )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            from ..logging_config import logger
            logger.warning(f"SetWindowDisplayAffinity not available: {e}")

    def set_error(self, text: str, *, transient: bool = False):
        if transient:
            self.status_lbl.setText(text)
            self.status_lbl.setStyleSheet("color: #FF9800; font-size: 12px;")
            self._transient_timer.start(_TRANSIENT_MS)
            return
        self._has_persistent_error = True
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(
            "color: #F44336; font-size: 12px; font-weight: bold;"
        )

    def clear_persistent_error(self):
        self._has_persistent_error = False
        self.status_lbl.setStyleSheet("color: #888888; font-size: 12px;")

    def _clear_transient_status(self):
        if self._has_persistent_error:
            return
        self.status_lbl.setText("")
        self.status_lbl.setStyleSheet("color: #888888; font-size: 12px;")

    def set_status(self, text: str):
        if self._has_persistent_error and not text:
            return
        if self._has_persistent_error:
            # Don't overwrite persistent error with routine status
            return
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet("color: #888888; font-size: 12px;")

    def populate_profiles(self, profiles: list[str], active: str | None):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(profiles)
        if active and active in profiles:
            self.profile_combo.setCurrentText(active)
        self.profile_combo.blockSignals(False)
        self.profile_combo.currentTextChanged.connect(self._profile_changed)

    def populate_devices(self, devices: list[dict], active_id: str | None):
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("Auto (default loopback)", None)
        selected = 0
        for i, d in enumerate(devices):
            label = d["name"]
            if d.get("is_default"):
                label += " (default)"
            if not d.get("is_loopback"):
                label += " [mic]"
            self.device_combo.addItem(label, d["id"])
            if active_id and d["id"] == active_id:
                selected = i + 1
        self.device_combo.setCurrentIndex(selected)
        self.device_combo.blockSignals(False)
        self.device_combo.currentIndexChanged.connect(self._device_changed)

    def _profile_changed(self, name: str):
        if self.on_profile_changed and name:
            self.on_profile_changed(name)

    def _device_changed(self, index: int):
        if self.on_device_changed:
            self.on_device_changed(self.device_combo.itemData(index))

    def _on_pause_clicked(self):
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")
        if self.on_pause_toggled:
            self.on_pause_toggled(self._paused)

    def clear_cards(self):
        for w in list(self._result_widgets):
            self.scroll_layout.removeWidget(w)
            w.deleteLater()
        self._result_widgets.clear()
        self._widget_map.clear()
        if self.on_clear_history:
            self.on_clear_history()

    def eventFilter(self, obj, event):
        if event.type() == event.Type.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self._drag_pos = event.globalPosition().toPoint()
        elif event.type() == event.Type.MouseMove:
            if self._drag_pos is not None:
                delta = event.globalPosition().toPoint() - self._drag_pos
                self.move(self.pos() + delta)
                self._drag_pos = event.globalPosition().toPoint()
                return True
        elif event.type() == event.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
            self._drag_pos = None
        return super().eventFilter(obj, event)

    def add_transcript(self, result: PipelineResult):
        widget = ResultWidget(result)
        self.scroll_layout.addWidget(widget)
        self._result_widgets.append(widget)
        self._widget_map[result.id] = widget

        while len(self._result_widgets) > _MAX_RESULT_CARDS:
            oldest = self._result_widgets.pop(0)
            self._widget_map.pop(oldest.phrase_id, None)
            self.scroll_layout.removeWidget(oldest)
            oldest.deleteLater()

        QTimer.singleShot(50, self._scroll_to_bottom)

    def update_suggestion(self, result: PipelineResult):
        widget = self._widget_map.get(result.id)
        if widget:
            if getattr(result, "translation_ru", None):
                widget.set_translation(result.translation_ru)
            widget.update_suggestion(result)
            if getattr(result, "answer_ru", None):
                widget.set_answer_ru(result.answer_ru)
            QTimer.singleShot(50, self._scroll_to_bottom)

    def append_token(self, phrase_id, token: str):
        widget = self._widget_map.get(phrase_id)
        if widget:
            widget.append_token(token)
            QTimer.singleShot(10, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _restore_geometry(self):
        path = get_window_state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            geo = data.get("geometry")
            if geo:
                self.setGeometry(geo["x"], geo["y"], geo["w"], geo["h"])
        except Exception:
            pass

    def save_geometry(self):
        path = get_window_state_path()
        try:
            g = self.geometry()
            path.write_text(
                json.dumps({"geometry": {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def closeEvent(self, event):
        self.save_geometry()
        super().closeEvent(event)
