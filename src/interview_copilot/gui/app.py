import asyncio
import sys
import threading

from PySide6.QtWidgets import QApplication

from ..audio.soundcard_wasapi import SoundCardWASAPIBackend
from ..logging_config import logger
from ..pipeline import InterviewPipeline
from ..suggestion.profile_manager import ProfileManager
from .main_window import CopilotMainWindow
from .signals import PipelineSignals


def start_gui(device_id: str | None = None):
    """Entry point for the GUI app."""
    from ..config import config
    device_id = device_id or config.AUDIO_DEVICE

    app = QApplication(sys.argv)

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

    window = CopilotMainWindow()
    window.set_status("⏳ Loading models...")
    window.show()

    # Populate profile/device lists early
    profile_mgr = ProfileManager()
    profiles = profile_mgr.list_profiles()
    active = None
    try:
        snap = profile_mgr.load_active_profile()
        if snap:
            active = snap.name
    except Exception:
        pass
    window.populate_profiles(profiles, active)

    try:
        devices = SoundCardWASAPIBackend().list_devices()
        window.populate_devices(devices, device_id)
    except Exception as e:
        logger.warning(f"Could not list audio devices: {e}")
        window.populate_devices([], device_id)

    signals.transcript_ready.connect(window.add_transcript)
    signals.suggestion_ready.connect(window.update_suggestion)
    signals.suggestion_token.connect(window.append_token)

    def on_error_signal(msg: str, transient: bool = False):
        window.set_error(msg, transient=transient)

    signals.error_occurred.connect(on_error_signal)
    signals.status_changed.connect(window.set_status)

    def on_audio_status(is_connected: bool, msg: str):
        if is_connected:
            window.clear_persistent_error()
            if not window._has_persistent_error:
                window.set_status("")
        else:
            window.set_status(msg)

    signals.audio_status_changed.connect(on_audio_status)

    pipeline_ref = []

    def run_pipeline():
        try:
            pipeline = InterviewPipeline(profile_mgr=profile_mgr)
            pipeline_ref.append(pipeline)

            signals.status_changed.emit("")

            def on_transcript(result):
                signals.transcript_ready.emit(result)

            def on_suggestion(result):
                signals.suggestion_ready.emit(result)

            def on_token(phrase_id, token):
                signals.suggestion_token.emit(phrase_id, token)

            def on_error(msg, transient=False):
                signals.error_occurred.emit(msg, transient)

            def on_audio_status(is_connected, msg):
                signals.audio_status_changed.emit(is_connected, msg)

            def on_status(msg):
                signals.status_changed.emit(msg)

            pipeline.set_callbacks(
                transcript_callback=on_transcript,
                suggestion_callback=on_suggestion,
                token_callback=on_token,
                error_callback=on_error,
                audio_status_callback=on_audio_status,
                status_callback=on_status,
            )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(pipeline.start(device_id))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Pipeline thread error: {e}")
            signals.status_changed.emit(f"❌ Error: {e}")

    def on_quit():
        window.save_geometry()
        if pipeline_ref:
            pipeline_ref[0].stop()

    app.aboutToQuit.connect(on_quit)

    def on_profile_changed(name: str):
        if pipeline_ref:
            ok = pipeline_ref[0].set_active_profile(name)
            if not ok:
                window.set_error(f"Failed to load profile: {name}", transient=True)
            else:
                window.clear_persistent_error()
                window.set_status(f"Profile: {name}")

    def on_device_changed(new_id):
        nonlocal device_id
        device_id = new_id
        if pipeline_ref:
            pipeline_ref[0].change_device(new_id)

    def on_pause_toggled(paused: bool):
        if not pipeline_ref:
            return
        if paused:
            pipeline_ref[0].pause_capture()
        else:
            pipeline_ref[0].resume_capture()

    def on_clear_history():
        if pipeline_ref:
            pipeline_ref[0].clear_history()

    window.on_profile_changed = on_profile_changed
    window.on_device_changed = on_device_changed
    window.on_pause_toggled = on_pause_toggled
    window.on_clear_history = on_clear_history

    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()

    exit_code = app.exec()

    pipeline_thread.join(timeout=3.0)

    if pipeline_thread.is_alive():
        import os
        logger.warning("Pipeline thread did not terminate in time. Forcing exit.")
        os._exit(exit_code)

    sys.exit(exit_code)


if __name__ == "__main__":
    start_gui()
