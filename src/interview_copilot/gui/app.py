import asyncio
import sys
import threading

from PySide6.QtWidgets import QApplication

from ..audio.soundcard_wasapi import SoundCardWASAPIBackend
from ..logging_config import logger
from ..pipeline import InterviewPipeline
from ..suggestion.profile_manager import ProfileManager
from . import theme
from .main_window import CopilotMainWindow
from .signals import PipelineSignals

_ERROR_SOURCES = (
    ("Audio device lost", "audio"),
    ("No candidate profile", "profile"),
    ("OpenRouter API key not set", "openrouter"),
)


def error_source(msg: str) -> str | None:
    """Tag pipeline errors so recovering one subsystem doesn't clear another's error."""
    for prefix, source in _ERROR_SOURCES:
        if msg.startswith(prefix):
            return source
    return None


def start_gui(device_id: str | None = None):
    """Entry point for the GUI app."""
    from ..config import config
    from ..settings_store import apply_prefs_to_config

    apply_prefs_to_config()
    device_id = device_id or CopilotMainWindow.load_persisted_device() or config.AUDIO_DEVICE

    app = QApplication(sys.argv)

    app.setStyleSheet(theme.app_stylesheet())

    signals = PipelineSignals()

    window = CopilotMainWindow()
    window.set_status("Loading models...")
    window.set_pipeline_ready(False)
    window.show()
    window.install_global_hotkey()
    window.setup_tray()

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

    signals.transcript_ready.connect(window.upsert_transcript)
    signals.transcript_removed.connect(window.remove_transcript)
    signals.suggestion_ready.connect(window.update_suggestion)
    signals.suggestion_token.connect(window.append_token)

    def on_error_signal(msg: str, transient: bool = False):
        window.set_error(msg, transient=transient, source=error_source(msg))

    signals.error_occurred.connect(on_error_signal)
    signals.status_changed.connect(window.set_status)

    def on_audio_status(is_connected: bool, msg: str):
        if is_connected:
            window.clear_persistent_error(source="audio")
            if not window.has_persistent_error:
                window.set_status("")
        else:
            window.set_error(msg, source="audio")

    signals.audio_status_changed.connect(on_audio_status)

    pipeline_ref = []
    pipeline_ready = False
    keys_reload_pending = False

    def run_pipeline():
        try:
            pipeline = InterviewPipeline(profile_mgr=profile_mgr)
            pipeline_ref.append(pipeline)

            signals.status_changed.emit("")

            def on_transcript(result):
                signals.transcript_ready.emit(result)

            def on_transcript_remove(phrase_id):
                signals.transcript_removed.emit(phrase_id)

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
                transcript_remove_callback=on_transcript_remove,
            )

            async def start_and_signal_ready():
                task = asyncio.ensure_future(pipeline.start(device_id))
                # start() clears the pause event; let it run before the GUI enables Pause.
                await asyncio.sleep(0)
                if not task.done():
                    signals.pipeline_ready.emit()
                await task

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(start_and_signal_ready())
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Pipeline thread error: {e}")
            signals.status_changed.emit(f"Error: {e}")

    def reload_api_keys():
        reload = getattr(pipeline_ref[0], "reload_api_keys", None)
        if reload is None:
            window.set_error("API keys saved. Restart to apply them.", transient=True)
            return
        try:
            reload()
        except Exception as e:
            logger.error(f"Failed to reload API keys: {e}")
            window.set_error(f"Failed to apply API keys: {e}", transient=True)
            return
        if config.OPENROUTER_API_KEY:
            window.clear_persistent_error(source="openrouter")

    def on_pipeline_ready():
        nonlocal pipeline_ready, keys_reload_pending
        pipeline_ready = True
        window.set_pipeline_ready(True)
        if keys_reload_pending:
            keys_reload_pending = False
            reload_api_keys()

    signals.pipeline_ready.connect(on_pipeline_ready)

    def on_settings_saved():
        nonlocal keys_reload_pending
        if pipeline_ready:
            reload_api_keys()
        else:
            keys_reload_pending = True

    def on_quit():
        window.save_geometry()
        window.release_global_hotkey()
        if pipeline_ref:
            pipeline_ref[0].stop()

    app.aboutToQuit.connect(on_quit)

    def on_profile_changed(name: str):
        if pipeline_ref:
            ok = pipeline_ref[0].set_active_profile(name)
            if not ok:
                window.set_error(f"Failed to load profile: {name}", transient=True)
            else:
                window.clear_persistent_error(source="profile")
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
    window.on_settings_saved = on_settings_saved

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
