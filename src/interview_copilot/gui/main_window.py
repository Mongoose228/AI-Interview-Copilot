import ctypes
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..models import PipelineResult

# Maximum number of result cards to keep in the overlay
_MAX_RESULT_CARDS = 50

# Windows constants for SetWindowDisplayAffinity
WDA_EXCLUDEFROMCAPTURE = 0x00000011


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

        # Original EN Text
        lbl_en = QLabel(f"🎙️ <b>EN:</b> {result.transcript}")
        lbl_en.setWordWrap(True)
        lbl_en.setStyleSheet("color: #CCCCCC; font-size: 13px;")
        layout.addWidget(lbl_en)

        # AI Suggestion Placeholder
        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line.setStyleSheet("background-color: rgba(100, 100, 100, 0.5);")
        layout.addWidget(self.line)

        self.lbl_ai = QLabel("🤔 <i>Thinking...</i>")
        self.lbl_ai.setWordWrap(True)
        self.lbl_ai.setStyleSheet("color: #888888; font-size: 13px; font-style: italic;")
        layout.addWidget(self.lbl_ai)
        
        self.current_suggestion_text = ""

    def append_token(self, token: str):
        if not self.current_suggestion_text:
            self.lbl_ai.setStyleSheet("color: #4CAF50; font-size: 15px; font-weight: bold;")
        self.current_suggestion_text += token
        self.lbl_ai.setText(f"💡 <b>AI:</b> {self.current_suggestion_text}")

    def update_suggestion(self, suggestion):
        if not suggestion:
            self.lbl_ai.setText("❌ <i>Suggestion failed or dropped.</i>")
            self.lbl_ai.setStyleSheet("color: #F44336; font-size: 13px; font-style: italic; border: 1px solid rgba(244, 67, 54, 0.5); padding: 4px; border-radius: 4px;")
            return
            
        text = suggestion.answer_en
        hedges = ["i'm not sure", "i am not sure", "i think", "might be", "could be", "maybe", "perhaps"]
        is_unsure = any(h in text.lower() for h in hedges)
        icon = "⚠️" if is_unsure else "✅"
        
        self.current_suggestion_text = text
        self.lbl_ai.setStyleSheet("color: #4CAF50; font-size: 15px; font-weight: bold;")
        self.lbl_ai.setText(f"💡 <b>AI {icon}:</b> {text}")


class CopilotMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Interview Copilot")
        self.resize(450, 600)

        # Make the window frameless and always on top
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Main central widget with border radius and semi-transparent background
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

        # Header (Drag handle + close button)
        self.header_layout = QHBoxLayout()
        self.title_lbl = QLabel("AI Interview Copilot")
        self.title_lbl.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        self.header_layout.addWidget(self.title_lbl)
        self.header_layout.addStretch()

        # Status label (shown during loading)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #888888; font-size: 12px;")
        self.header_layout.addWidget(self.status_lbl)

        # Close button
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
        from PySide6.QtWidgets import QApplication
        self.close_btn.clicked.connect(QApplication.instance().quit)
        self.header_layout.addWidget(self.close_btn)

        self.main_layout.addLayout(self.header_layout)

        # Scroll Area for results
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setContentsMargins(0, 0, 10, 0)
        self.scroll_area.setWidget(self.scroll_content)

        self.main_layout.addWidget(self.scroll_area)

        # For dragging the frameless window
        self._drag_pos = None

        # Track result widgets for ring buffer eviction and ID mapping
        self._result_widgets: list[ResultWidget] = []
        self._widget_map = {}

        # Keyboard shortcut: Escape to close
        shortcut_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut_esc.activated.connect(QApplication.instance().quit)

        # Install event filter on header widgets to enable dragging
        self.title_lbl.installEventFilter(self)
        self.status_lbl.installEventFilter(self)

    def show(self):
        super().show()
        self._apply_display_affinity()

    def _apply_display_affinity(self):
        """Hide this window from screen capture (OBS, Zoom, etc.) on Windows."""
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            # WDA_EXCLUDEFROMCAPTURE = 0x11 (Windows 10 2004+)
            result = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            if result:
                from ..logging_config import logger
                logger.info("Window display affinity set: excluded from screen capture.")
            else:
                from ..logging_config import logger
                logger.warning(
                    "Failed to set window display affinity. "
                    "Window may be visible in screen capture."
                )
        except Exception as e:
            from ..logging_config import logger
            logger.warning(f"SetWindowDisplayAffinity not available: {e}")

    def set_status(self, text: str):
        """Update the status label in the header."""
        self.status_lbl.setText(text)

    def eventFilter(self, obj, event):
        if event.type() == event.Type.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self._drag_pos = event.globalPosition().toPoint()
        elif event.type() == event.Type.MouseMove:
            if self._drag_pos is not None:
                delta = event.globalPosition().toPoint() - self._drag_pos
                self.move(self.pos() + delta)
                self._drag_pos = event.globalPosition().toPoint()
                return True # Consume to prevent text selection while dragging
        elif event.type() == event.Type.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self._drag_pos = None
        return super().eventFilter(obj, event)

    def add_transcript(self, result: PipelineResult):
        """Called via signal when a new transcript is ready."""
        widget = ResultWidget(result)
        self.scroll_layout.addWidget(widget)
        self._result_widgets.append(widget)
        self._widget_map[result.id] = widget

        # Ring buffer: evict oldest cards when exceeding limit
        while len(self._result_widgets) > _MAX_RESULT_CARDS:
            oldest = self._result_widgets.pop(0)
            self._widget_map.pop(oldest.phrase_id, None)
            self.scroll_layout.removeWidget(oldest)
            oldest.deleteLater()

        # Auto-scroll to bottom
        QTimer.singleShot(50, self._scroll_to_bottom)

    def update_suggestion(self, result: PipelineResult):
        widget = self._widget_map.get(result.id)
        if widget:
            widget.update_suggestion(result.suggestion)
            QTimer.singleShot(50, self._scroll_to_bottom)

    def append_token(self, phrase_id, token: str):
        widget = self._widget_map.get(phrase_id)
        if widget:
            widget.append_token(token)
            QTimer.singleShot(10, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
