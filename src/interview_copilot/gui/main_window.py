from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..models import PipelineResult


class ResultWidget(QFrame):
    """A widget to display a single pipeline result."""

    def __init__(self, result: PipelineResult, parent=None):
        super().__init__(parent)
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
        lbl_en = QLabel(f"🗣️ <b>EN:</b> {result.transcript}")
        lbl_en.setWordWrap(True)
        lbl_en.setStyleSheet("color: #CCCCCC; font-size: 13px;")
        layout.addWidget(lbl_en)

        # Translation RU Text
        if result.translation_ru:
            lbl_ru = QLabel(f"🇷🇺 <b>RU:</b> {result.translation_ru}")
            lbl_ru.setWordWrap(True)
            lbl_ru.setStyleSheet("color: #AAAAAA; font-size: 13px; font-style: italic;")
            layout.addWidget(lbl_ru)

        # AI Suggestion
        if result.suggestion:
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setStyleSheet("background-color: rgba(100, 100, 100, 0.5);")
            layout.addWidget(line)

            verify_icon = "⚠️" if result.suggestion.needs_verification else "✅"
            lbl_ai = QLabel(f"💡 <b>AI {verify_icon}:</b> {result.suggestion.answer_ru}")
            lbl_ai.setWordWrap(True)
            lbl_ai.setStyleSheet("color: #4CAF50; font-size: 15px; font-weight: bold;")
            layout.addWidget(lbl_ai)


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

        # Header (Drag handle)
        self.header_layout = QHBoxLayout()
        self.title_lbl = QLabel("AI Interview Copilot")
        self.title_lbl.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        self.header_layout.addWidget(self.title_lbl)
        self.header_layout.addStretch()

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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None

    def add_result(self, result: PipelineResult):
        """Called via signal when a new result is ready."""
        widget = ResultWidget(result)
        self.scroll_layout.addWidget(widget)

        # Auto-scroll to bottom
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
