import asyncio
import sys
import threading

from PySide6.QtWidgets import QApplication

from ..config import config
from ..logging_config import logger
from ..pipeline import InterviewPipeline
from .main_window import CopilotMainWindow
from .signals import PipelineSignals


def start_gui(device_id: str = None):
    """Entry point for the GUI app."""
    app = QApplication(sys.argv)

    # Global stylesheet for the app
    app.setStyleSheet("""
        QWidget {
            background-color: #1E1E1E;
            color: #E0E0E0;
            font-family: 'Segoe UI', Inter, sans-serif;
            font-size: 14px;
        }
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QScrollBar:vertical {
            border: none;
            background: #2C2C2C;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #555555;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
    """)

    signals = PipelineSignals()

    # Initialize pipeline
    pipeline = InterviewPipeline()

    # Provide a callback to the pipeline that emits the Qt signal
    def on_result(result):
        signals.result_ready.emit(result)

    pipeline.set_result_callback(on_result)

    # Run the pipeline in a separate thread so it doesn't block the Qt Event Loop
    def run_pipeline():
        # new event loop for this thread because pipeline uses asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(pipeline.start(device_id))
        except Exception as e:
            logger.error(f"Pipeline thread error: {e}")
        finally:
            loop.close()

    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()

    # Create and show window
    window = CopilotMainWindow()

    # Connect signals
    signals.result_ready.connect(window.add_result)

    window.show()

    # Start Qt Event Loop
    exit_code = app.exec()

    # Cleanup
    pipeline.stop()
    pipeline_thread.join(timeout=2.0)
    sys.exit(exit_code)
