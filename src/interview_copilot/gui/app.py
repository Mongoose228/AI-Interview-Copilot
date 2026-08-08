import asyncio
import sys
import threading

from PySide6.QtWidgets import QApplication

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
            color: #E0E0E0;
            font-family: 'Segoe UI', Inter, sans-serif;
            font-size: 14px;
            background: transparent;
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

    # Create and show window IMMEDIATELY — before pipeline init
    window = CopilotMainWindow()
    window.set_status("⏳ Loading models...")
    window.show()

    # Connect signals
    signals.transcript_ready.connect(window.add_transcript)
    signals.suggestion_ready.connect(window.update_suggestion)
    signals.suggestion_token.connect(window.append_token)
    signals.error_occurred.connect(
        lambda msg: window.set_status(f"⚠️ {msg}")
    )
    signals.status_changed.connect(window.set_status)
    signals.audio_status_changed.connect(
        lambda is_connected, msg: window.set_status(msg if not is_connected else "")
    )

    # Initialize and run pipeline in a background thread
    # This prevents "Not Responding" while Whisper downloads/warms up
    pipeline_ref = []

    def run_pipeline():
        try:
            # Heavy initialization happens here, in the background thread
            pipeline = InterviewPipeline()
            pipeline_ref.append(pipeline)

            # Signal that loading is complete
            signals.status_changed.emit("")

            # Provide a callback to the pipeline that emits the Qt signal
            def on_transcript(result):
                signals.transcript_ready.emit(result)

            def on_suggestion(result):
                signals.suggestion_ready.emit(result)

            def on_token(phrase_id, token):
                signals.suggestion_token.emit(phrase_id, token)
            
            def on_error(msg):
                signals.error_occurred.emit(msg)

            def on_audio_status(is_connected, msg):
                signals.audio_status_changed.emit(is_connected, msg)

            pipeline.set_callbacks(
                transcript_callback=on_transcript,
                suggestion_callback=on_suggestion,
                token_callback=on_token,
                error_callback=on_error,
                audio_status_callback=on_audio_status
            )

            # new event loop for this thread because pipeline uses asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(pipeline.start(device_id))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Pipeline thread error: {e}")
            signals.status_changed.emit(f"❌ Error: {e}")

    # Set up instantaneous shutdown
    def on_quit():
        # Force the OS to instantly kill the process and all blocked background threads.
        import os
        os._exit(0)

    app.aboutToQuit.connect(on_quit)

    pipeline_thread = threading.Thread(target=run_pipeline, daemon=False)
    pipeline_thread.start()

    # Start Qt Event Loop
    exit_code = app.exec()

    # Wait for the pipeline thread to gracefully exit
    pipeline_thread.join(timeout=3.0)

    import os
    os._exit(exit_code)

if __name__ == "__main__":
    start_gui()
