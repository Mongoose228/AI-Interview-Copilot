import ctypes
import html
import json
import sys

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from ..logging_config import logger
from ..models import PipelineResult, SuggestionStatus
from ..paths import get_window_state_path
from . import theme
from .privacy_banner import PrivacyBanner

_MAX_RESULT_CARDS = 50
_TRANSIENT_MS = 7000
_PROFILE_PLACEHOLDER = "— select profile —"
WDA_EXCLUDEFROMCAPTURE = 0x00000011
_HOTKEY_ID = 0xC0B1
_MOD_CONTROL = 0x0002
_MOD_SHIFT = 0x0004
_MOD_NOREPEAT = 0x4000
_WM_HOTKEY = 0x0312


def _esc(text: str) -> str:
    return html.escape(text or "", quote=True)


class ResultWidget(QFrame):
    """A widget to display a single pipeline result."""

    def __init__(self, result: PipelineResult, parent=None):
        super().__init__(parent)
        self.phrase_id = result.id
        self.setObjectName("ResultWidget")
        self.setStyleSheet(theme.result_card_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(6)

        self.lbl_en = QLabel()
        self.lbl_en.setWordWrap(True)
        self.lbl_en.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_en)

        self.lbl_ru = QLabel("")
        self.lbl_ru.setWordWrap(True)
        self.lbl_ru.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_ru.setStyleSheet(f"color: {theme.TEXT_RU}; font-size: 12px;")
        self.lbl_ru.setVisible(False)
        layout.addWidget(self.lbl_ru)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setStyleSheet(theme.divider_style())
        layout.addWidget(self.line)

        self.lbl_ai = QLabel("<i>Thinking...</i>")
        self.lbl_ai.setWordWrap(True)
        self.lbl_ai.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_ai.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 13px; font-style: italic;"
        )
        layout.addWidget(self.lbl_ai)

        self.lbl_ai_ru = QLabel("")
        self.lbl_ai_ru.setWordWrap(True)
        self.lbl_ai_ru.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_ai_ru.setStyleSheet(f"color: {theme.TEXT_AI_RU}; font-size: 13px;")
        self.lbl_ai_ru.setVisible(False)
        layout.addWidget(self.lbl_ai_ru)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 2, 0, 0)
        btn_row.addStretch()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setFixedHeight(24)
        self.copy_btn.setCursor(Qt.PointingHandCursor)
        self.copy_btn.setToolTip("Copy suggestion to clipboard")
        self.copy_btn.setStyleSheet(theme.copy_button_style())
        self.copy_btn.clicked.connect(self._copy_suggestion)
        self.copy_btn.setVisible(False)
        btn_row.addWidget(self.copy_btn)
        layout.addLayout(btn_row)

        self.current_suggestion_text = ""
        self.update_transcript(result)
        self._fade_in()

    def _fade_in(self) -> None:
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(220)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        # A lingering effect forces offscreen rendering of every card on repaint.
        anim.finished.connect(lambda: self.setGraphicsEffect(None))
        anim.start()
        self._fade_anim = anim

    def _set_ai_text(self, body: str, *, suffix: str = "") -> None:
        self.lbl_ai.setText(f"{theme.tag_html('AI', theme.ACCENT)}{suffix}&nbsp; {body}")

    def update_transcript(self, result: PipelineResult):
        """Update provisional/final transcript without replacing the card."""
        self.lbl_en.setText(f"{theme.tag_html('EN')}&nbsp; {_esc(result.transcript)}")
        if result.is_provisional:
            self.lbl_en.setStyleSheet(
                f"color: {theme.TEXT_PROVISIONAL}; font-size: 13px; font-style: italic;"
            )
            if not self.current_suggestion_text:
                self.lbl_ai.setText("<i>Listening...</i>")
                self.lbl_ai.setStyleSheet(
                    f"color: {theme.TEXT_MUTED}; font-size: 13px; font-style: italic;"
                )
        else:
            self.lbl_en.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 13px;")
            if not self.current_suggestion_text:
                self.lbl_ai.setText("<i>Thinking...</i>")
                self.lbl_ai.setStyleSheet(
                    f"color: {theme.TEXT_MUTED}; font-size: 13px; font-style: italic;"
                )

    def _copy_suggestion(self):
        if self.current_suggestion_text:
            QApplication.clipboard().setText(self.current_suggestion_text)
            self.copy_btn.setText("Copied")
            QTimer.singleShot(1200, lambda: self.copy_btn.setText("Copy"))

    def append_token(self, token: str):
        if not self.current_suggestion_text:
            self.lbl_ai.setStyleSheet(
                f"color: {theme.TEXT_AI_BOLD}; font-size: 14px; font-weight: 600;"
            )
        self.current_suggestion_text += token
        formatted_text = _esc(self.current_suggestion_text).replace("\n", "<br>")
        self._set_ai_text(formatted_text)
        self.copy_btn.setVisible(True)

    def update_suggestion(self, result: PipelineResult):
        status = getattr(result, "suggestion_status", SuggestionStatus.PENDING)
        suggestion = result.suggestion
        is_cancelled = getattr(result, "is_cancelled", False)

        if status == SuggestionStatus.INTERRUPTED or (
            is_cancelled and self.current_suggestion_text
        ):
            formatted = _esc(self.current_suggestion_text).replace("\n", "<br>")
            suffix = " <i>(interrupted)</i>" if self.current_suggestion_text else ""
            self._set_ai_text(f"{formatted}{suffix}")
            self.lbl_ai.setStyleSheet(f"color: {theme.TEXT_WARN}; font-size: 13px;")
            return

        if status == SuggestionStatus.CANCELLED or is_cancelled:
            self.lbl_ai.setText("<i>Skipped (new question)</i>")
            self.lbl_ai.setStyleSheet(
                f"color: {theme.TEXT_MUTED}; font-size: 12px; font-style: italic;"
                f" border: 1px solid {theme.BORDER_SUBTLE};"
                f" padding: 4px 6px; border-radius: {theme.RADIUS_SM};"
            )
            return

        if status == SuggestionStatus.SKIPPED_NO_KEY:
            self.lbl_ai.setText(
                "<i>Suggestions disabled — set OPENROUTER_API_KEY</i>"
            )
            self.lbl_ai.setStyleSheet(
                f"color: {theme.TEXT_WARN}; font-size: 12px; font-style: italic;"
            )
            return

        if status == SuggestionStatus.SKIPPED_NO_PROFILE:
            self.lbl_ai.setText(
                "<i>Suggestions disabled — create a candidate profile</i>"
            )
            self.lbl_ai.setStyleSheet(
                f"color: {theme.TEXT_WARN}; font-size: 12px; font-style: italic;"
            )
            return

        if not suggestion or status == SuggestionStatus.FAILED:
            self.lbl_ai.setText("<i>Suggestion failed.</i>")
            self.lbl_ai.setStyleSheet(
                f"color: {theme.TEXT_ERROR}; font-size: 12px; font-style: italic;"
                f" border: 1px solid rgba(232, 106, 93, 0.45);"
                f" padding: 4px 6px; border-radius: {theme.RADIUS_SM};"
            )
            return

        text = suggestion.answer_en
        hedge = " " + theme.tag_html("hedged", theme.TEXT_WARN)
        self.current_suggestion_text = text
        formatted_text = _esc(text).replace("\n", "<br>")
        self.lbl_ai.setStyleSheet(
            f"color: {theme.TEXT_AI_BOLD}; font-size: 14px; font-weight: 600;"
        )
        self._set_ai_text(formatted_text, suffix=hedge if suggestion.has_hedging else "")
        self.copy_btn.setVisible(bool(text))

    def set_translation(self, translation_ru: str):
        if translation_ru:
            self.lbl_ru.setText(f"{theme.tag_html('RU')}&nbsp; {_esc(translation_ru)}")
            self.lbl_ru.setVisible(True)

    def set_answer_ru(self, answer_ru: str):
        if answer_ru:
            formatted = _esc(answer_ru).replace("\n", "<br>")
            self.lbl_ai_ru.setText(f"{theme.tag_html('AI RU')}&nbsp; {formatted}")
            self.lbl_ai_ru.setVisible(True)


class CopilotMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Interview Copilot")
        self.resize(460, 640)
        self.setMinimumSize(340, 300)

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.central_widget = QWidget()
        self.central_widget.setStyleSheet(theme.main_container_style())
        self.central_widget.setObjectName("MainContainer")
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(16, 14, 16, 12)
        self.main_layout.setSpacing(10)

        # Brand / header
        self.header_layout = QHBoxLayout()
        self.header_layout.setSpacing(8)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(1)
        self.title_lbl = QLabel("AI Interview Copilot")
        self.title_lbl.setStyleSheet(theme.title_style())
        brand_col.addWidget(self.title_lbl)
        self.subtitle_lbl = QLabel("Live suggestions")
        self.subtitle_lbl.setStyleSheet(theme.caption_style())
        brand_col.addWidget(self.subtitle_lbl)
        self.header_layout.addLayout(brand_col)
        self.header_layout.addStretch()

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setFixedHeight(30)
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.setStyleSheet(theme.button_style())
        self.header_layout.addWidget(self.pause_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedHeight(30)
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.setStyleSheet(theme.button_style())
        self.header_layout.addWidget(self.clear_btn)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setToolTip("Quit (Esc hides, Ctrl+Shift+H shows/hides)")
        self.close_btn.setStyleSheet(theme.close_button_style())
        self.close_btn.clicked.connect(QApplication.instance().quit)
        self.header_layout.addWidget(self.close_btn)
        self.main_layout.addLayout(self.header_layout)

        # Controls: profile + device
        controls = QHBoxLayout()
        controls.setSpacing(8)

        profile_box = QVBoxLayout()
        profile_box.setSpacing(3)
        profile_lbl = QLabel("PROFILE")
        profile_lbl.setStyleSheet(theme.caption_style())
        profile_box.addWidget(profile_lbl)
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(120)
        self.profile_combo.setStyleSheet(theme.combo_style())
        profile_box.addWidget(self.profile_combo)
        controls.addLayout(profile_box, 1)

        device_box = QVBoxLayout()
        device_box.setSpacing(3)
        device_lbl = QLabel("DEVICE")
        device_lbl.setStyleSheet(theme.caption_style())
        device_box.addWidget(device_lbl)
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(140)
        self.device_combo.setStyleSheet(theme.combo_style())
        device_box.addWidget(self.device_combo)
        controls.addLayout(device_box, 1)

        settings_col = QVBoxLayout()
        settings_col.setSpacing(3)
        settings_spacer = QLabel(" ")
        settings_spacer.setStyleSheet(theme.caption_style())
        settings_col.addWidget(settings_spacer)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setFixedHeight(30)
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setStyleSheet(theme.button_style())
        self.settings_btn.clicked.connect(self._open_settings)
        settings_col.addWidget(self.settings_btn)
        controls.addLayout(settings_col)

        self.main_layout.addLayout(controls)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(theme.status_style_idle())
        self.status_lbl.setVisible(False)
        self.main_layout.addWidget(self.status_lbl)

        self.privacy_banner = PrivacyBanner()
        self.main_layout.addWidget(self.privacy_banner)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFocusPolicy(Qt.NoFocus)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setContentsMargins(0, 4, 6, 0)
        self.scroll_layout.setSpacing(0)

        self.empty_lbl = QLabel(
            "Waiting for interview audio.\n"
            "Transcripts and AI answers will appear here."
        )
        self.empty_lbl.setWordWrap(True)
        self.empty_lbl.setAlignment(Qt.AlignCenter)
        self.empty_lbl.setStyleSheet(theme.empty_state_style())
        self.scroll_layout.addWidget(self.empty_lbl)

        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area, 1)

        grip_row = QHBoxLayout()
        grip_row.addStretch()
        self.size_grip = QSizeGrip(self.central_widget)
        grip_row.addWidget(self.size_grip, 0, Qt.AlignBottom | Qt.AlignRight)
        self.main_layout.addLayout(grip_row)

        self._drag_pos = None
        self._result_widgets: list[ResultWidget] = []
        self._widget_map = {}
        # source -> message; the most recently set one is displayed.
        self._persistent_errors: dict[str | None, str] = {}
        self._paused = False
        self._transient_timer = QTimer(self)
        self._transient_timer.setSingleShot(True)
        self._transient_timer.timeout.connect(self._clear_transient_status)
        self._hotkey_hwnd: int | None = None
        self._tray: QSystemTrayIcon | None = None

        shortcut_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut_esc.activated.connect(self.hide_to_background)

        self._hide_shortcut = QShortcut(QKeySequence("Ctrl+Shift+H"), self)
        self._hide_shortcut.activated.connect(self.toggle_visibility)

        self.title_lbl.installEventFilter(self)
        self.subtitle_lbl.installEventFilter(self)
        self.status_lbl.installEventFilter(self)

        self.clear_btn.clicked.connect(self.clear_cards)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.profile_combo.currentIndexChanged.connect(self._profile_changed)
        self.device_combo.currentIndexChanged.connect(self._device_changed)

        # Callbacks wired by app.py
        self.on_profile_changed = None
        self.on_device_changed = None
        self.on_pause_toggled = None
        self.on_clear_history = None
        self.on_settings_saved = None

        self._restore_geometry()

    @property
    def has_persistent_error(self) -> bool:
        return bool(self._persistent_errors)

    def set_pipeline_ready(self, ready: bool) -> None:
        for w in (self.pause_btn, self.profile_combo, self.device_combo):
            w.setEnabled(ready)

    def _refresh_empty_state(self) -> None:
        self.empty_lbl.setVisible(len(self._result_widgets) == 0)

    def show(self):
        super().show()
        self._apply_display_affinity()

    def toggle_visibility(self):
        if self.isVisible():
            self.hide_to_background()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _can_restore_hidden(self) -> bool:
        return self._hotkey_hwnd is not None or (
            self._tray is not None and self._tray.isVisible()
        )

    def hide_to_background(self):
        # The frameless Qt.Tool window has no taskbar entry; never hide it unrecoverably.
        if not self._can_restore_hidden():
            self.set_error(
                "Can't hide: no global hotkey or tray icon available.", transient=True
            )
            return
        self.hide()

    def install_global_hotkey(self) -> bool:
        """Register Ctrl+Shift+H system-wide so the hidden window can be restored."""
        if sys.platform != "win32" or QGuiApplication.platformName() != "windows":
            return False
        try:
            hwnd = int(self.winId())
            ok = ctypes.windll.user32.RegisterHotKey(
                ctypes.c_void_p(hwnd),
                _HOTKEY_ID,
                _MOD_CONTROL | _MOD_SHIFT | _MOD_NOREPEAT,
                ord("H"),
            )
        except (AttributeError, OSError, ValueError) as e:
            logger.warning(f"Global hotkey unavailable: {e}")
            return False
        if not ok:
            logger.warning(
                "Failed to register global Ctrl+Shift+H hotkey "
                "(already in use?). Falling back to in-window shortcut."
            )
            return False
        self._hotkey_hwnd = hwnd
        self._hide_shortcut.setEnabled(False)
        return True

    def release_global_hotkey(self) -> None:
        if self._hotkey_hwnd is None:
            return
        try:
            ctypes.windll.user32.UnregisterHotKey(
                ctypes.c_void_p(self._hotkey_hwnd), _HOTKEY_ID
            )
        except (AttributeError, OSError) as e:
            logger.debug(f"UnregisterHotKey failed: {e}")
        self._hotkey_hwnd = None
        self._hide_shortcut.setEnabled(True)

    def nativeEvent(self, event_type, message):
        if self._hotkey_hwnd is not None:
            from ctypes import wintypes

            msg = wintypes.MSG.from_address(int(message))
            if msg.message == _WM_HOTKEY and msg.wParam == _HOTKEY_ID:
                self.toggle_visibility()
                return True, 0
        return super().nativeEvent(event_type, message)

    def setup_tray(self) -> bool:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return False
        icon = self.windowIcon()
        if icon.isNull():
            icon = QApplication.style().standardIcon(QStyle.SP_ComputerIcon)
        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip("AI Interview Copilot")
        menu = QMenu(self)
        menu.addAction("Show / Hide", self.toggle_visibility)
        menu.addSeparator()
        menu.addAction("Quit", QApplication.instance().quit)
        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        self._tray = tray
        return True

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            self.toggle_visibility()

    def _apply_display_affinity(self):
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            result = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            if result:
                logger.info("Window display affinity set: excluded from screen capture.")
            else:
                logger.warning(
                    "Failed to set window display affinity. "
                    "Window may be visible in screen capture."
                )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.warning(f"SetWindowDisplayAffinity not available: {e}")

    def _show_status(self, text: str, style: str) -> None:
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(style)
        self.status_lbl.setVisible(bool(text))

    def _render_persistent(self) -> None:
        if self._persistent_errors:
            text = next(reversed(self._persistent_errors.values()))
            self._show_status(text, theme.status_style_error())
        else:
            self._show_status("", theme.status_style_idle())

    def set_error(self, text: str, *, transient: bool = False, source: str | None = None):
        if transient:
            self._show_status(text, theme.status_style_warn())
            self._transient_timer.start(_TRANSIENT_MS)
            return
        self._persistent_errors.pop(source, None)
        self._persistent_errors[source] = text
        self._transient_timer.stop()
        self._render_persistent()

    def clear_persistent_error(self, source: str | None = None):
        """Clear persistent errors from ``source`` (all errors when None)."""
        if source is None:
            removed = bool(self._persistent_errors)
            self._persistent_errors.clear()
        else:
            removed = self._persistent_errors.pop(source, None) is not None
        if removed and not self._transient_timer.isActive():
            self._render_persistent()

    def _clear_transient_status(self):
        self._render_persistent()

    def set_status(self, text: str):
        if self._persistent_errors:
            # Don't overwrite persistent error with routine status
            return
        self._show_status(text, theme.status_style_idle())

    def populate_profiles(self, profiles: list[str], active: str | None):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        if not active or active not in profiles:
            # Without a real current item, picking the first profile wouldn't emit a change.
            self.profile_combo.addItem(_PROFILE_PLACEHOLDER, None)
        for name in profiles:
            self.profile_combo.addItem(name, name)
        if active and active in profiles:
            self.profile_combo.setCurrentIndex(self.profile_combo.findData(active))
        self.profile_combo.blockSignals(False)

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
        self.device_combo.setToolTip(self.device_combo.currentText())
        self.device_combo.blockSignals(False)

    def _profile_changed(self, index: int):
        name = self.profile_combo.itemData(index)
        if self.on_profile_changed and name:
            self.on_profile_changed(name)

    def _device_changed(self, index: int):
        device_id = self.device_combo.itemData(index)
        self.device_combo.setToolTip(self.device_combo.currentText())
        self._persist_device(device_id)
        if self.on_device_changed:
            self.on_device_changed(device_id)

    def _open_settings(self):
        from .settings_dialog import SettingsDialog

        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.Accepted and self.on_settings_saved:
            self.on_settings_saved()

    @staticmethod
    def _persist_device(device_id: str | None) -> None:
        path = get_window_state_path()
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data["audio_device"] = device_id
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def load_persisted_device() -> str | None:
        path = get_window_state_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("audio_device")
        except Exception:
            return None

    def _on_pause_clicked(self):
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")
        self.pause_btn.setStyleSheet(
            theme.button_style(primary=self._paused)
        )
        if self.on_pause_toggled:
            self.on_pause_toggled(self._paused)

    def clear_cards(self):
        for w in list(self._result_widgets):
            self.scroll_layout.removeWidget(w)
            w.deleteLater()
        self._result_widgets.clear()
        self._widget_map.clear()
        self._refresh_empty_state()
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

    def upsert_transcript(self, result: PipelineResult):
        existing = self._widget_map.get(result.id)
        if existing:
            existing.update_transcript(result)
            QTimer.singleShot(10, self._scroll_to_bottom)
            return

        widget = ResultWidget(result)
        self.scroll_layout.addWidget(widget)
        self._result_widgets.append(widget)
        self._widget_map[result.id] = widget
        self._refresh_empty_state()

        while len(self._result_widgets) > _MAX_RESULT_CARDS:
            oldest = self._result_widgets.pop(0)
            self._widget_map.pop(oldest.phrase_id, None)
            self.scroll_layout.removeWidget(oldest)
            oldest.deleteLater()

        QTimer.singleShot(50, self._scroll_to_bottom)

    # Backwards-compatible alias for callers/tests using the old API.
    def add_transcript(self, result: PipelineResult):
        self.upsert_transcript(result)

    def remove_transcript(self, phrase_id):
        widget = self._widget_map.pop(phrase_id, None)
        if not widget:
            return
        if widget in self._result_widgets:
            self._result_widgets.remove(widget)
        self.scroll_layout.removeWidget(widget)
        widget.deleteLater()
        self._refresh_empty_state()

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
            data = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            g = self.geometry()
            data["geometry"] = {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    def closeEvent(self, event):
        self.save_geometry()
        self.release_global_hotkey()
        if self._tray is not None:
            self._tray.hide()
        super().closeEvent(event)
