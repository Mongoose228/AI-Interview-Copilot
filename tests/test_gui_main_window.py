import ctypes
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from interview_copilot.config import config  # noqa: E402
from interview_copilot.gui import main_window as mw  # noqa: E402
from interview_copilot.gui import settings_dialog as sd  # noqa: E402

app = QApplication.instance() or QApplication([])


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "get_window_state_path", lambda: tmp_path / "window_state.json")
    w = mw.CopilotMainWindow()
    yield w
    w._hotkey_hwnd = None
    w.close()
    w.deleteLater()


def test_transient_error_restores_persistent(window):
    window.set_error("No API key")
    window.set_error("Network blip", transient=True)
    assert window.status_lbl.text() == "Network blip"

    window._transient_timer.stop()
    window._clear_transient_status()
    assert window.status_lbl.text() == "No API key"
    assert window.has_persistent_error


def test_transient_without_persistent_clears(window):
    window.set_error("Network blip", transient=True)
    window._clear_transient_status()
    assert window.status_lbl.text() == ""
    assert not window.status_lbl.isVisibleTo(window)


def test_audio_clear_keeps_unrelated_error(window):
    window.set_error("OpenRouter API key not set.", source="openrouter")
    window.set_error("Reconnecting... (1/5)", source="audio")
    assert window.status_lbl.text() == "Reconnecting... (1/5)"

    window.clear_persistent_error(source="audio")
    assert window.status_lbl.text() == "OpenRouter API key not set."
    assert window.has_persistent_error

    window.clear_persistent_error(source="audio")
    assert window.status_lbl.text() == "OpenRouter API key not set."

    window.clear_persistent_error()
    assert not window.has_persistent_error
    assert window.status_lbl.text() == ""


def test_status_does_not_override_persistent_error(window):
    window.set_error("Audio device lost", source="audio")
    window.set_status("Listening…")
    assert window.status_lbl.text() == "Audio device lost"


def test_clear_persistent_keeps_active_transient(window):
    window.set_error("Audio device lost", source="audio")
    window.set_error("Task error", transient=True)
    window.clear_persistent_error(source="audio")
    assert window.status_lbl.text() == "Task error"


def test_profile_placeholder_when_no_active(window):
    calls = []
    window.on_profile_changed = calls.append
    window.populate_profiles(["example_profile"], None)

    assert window.profile_combo.count() == 2
    assert window.profile_combo.itemData(0) is None
    assert window.profile_combo.currentIndex() == 0

    window.profile_combo.setCurrentIndex(1)
    assert calls == ["example_profile"]

    window.profile_combo.setCurrentIndex(0)
    assert calls == ["example_profile"]


def test_profile_active_has_no_placeholder(window):
    window.populate_profiles(["a", "b"], "b")
    assert window.profile_combo.count() == 2
    assert window.profile_combo.currentData() == "b"


def test_repopulate_does_not_duplicate_signals(window):
    profiles, devices = [], []
    window.on_profile_changed = profiles.append
    window.on_device_changed = devices.append
    devs = [{"id": "d1", "name": "Speakers", "is_loopback": True}]
    for _ in range(3):
        window.populate_profiles(["a", "b"], "a")
        window.populate_devices(devs, None)

    window.profile_combo.setCurrentIndex(1)
    window.device_combo.setCurrentIndex(1)
    assert profiles == ["b"]
    assert devices == ["d1"]


def test_controls_disabled_until_pipeline_ready(window):
    window.set_pipeline_ready(False)
    assert not window.pause_btn.isEnabled()
    assert not window.profile_combo.isEnabled()
    assert not window.device_combo.isEnabled()
    window.set_pipeline_ready(True)
    assert window.pause_btn.isEnabled()


def test_hide_refused_without_recovery_path(window, monkeypatch):
    hidden = []
    monkeypatch.setattr(window, "hide", lambda: hidden.append(True))
    window.hide_to_background()
    assert hidden == []
    assert "Can't hide" in window.status_lbl.text()


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 MSG layout")
def test_native_hotkey_toggles_visibility(window, monkeypatch):
    from ctypes import wintypes

    toggled = []
    monkeypatch.setattr(window, "toggle_visibility", lambda: toggled.append(True))
    window._hotkey_hwnd = 1
    msg = wintypes.MSG()
    msg.message = mw._WM_HOTKEY
    msg.wParam = mw._HOTKEY_ID
    handled = window.nativeEvent(b"windows_generic_MSG", ctypes.addressof(msg))
    assert handled[0] is True
    assert toggled == [True]


def test_settings_saved_callback_runs_on_accept(window, monkeypatch):
    monkeypatch.setattr(sd, "load_prefs", lambda: {"secrets": {}, "settings": {}})
    monkeypatch.setattr(sd.SettingsDialog, "exec", lambda self: QDialog.Accepted)
    calls = []
    window.on_settings_saved = lambda: calls.append(True)
    window._open_settings()
    assert calls == [True]


def test_settings_dialog_save_clears_and_sets_keys(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "SONIOX_API_KEY",
        "STT_BACKEND",
        "STT_DUAL_PATH_ENABLED",
        "AUDIO_DUAL_CHANNEL",
        "AUDIO_MIC_DEVICE",
        "DSP_DENOISE_ENABLED",
    ):
        monkeypatch.setattr(config, key, getattr(config, key))
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "old-key")
    saved = []
    monkeypatch.setattr(sd, "load_prefs", lambda: {"secrets": {}, "settings": {}})
    monkeypatch.setattr(sd, "save_prefs", saved.append)

    dlg = sd.SettingsDialog()
    assert dlg.openrouter.text() == "old-key"
    dlg.openrouter.setText("")
    dlg.soniox.setText("sx-123")
    dlg._save()

    assert config.OPENROUTER_API_KEY is None
    assert config.SONIOX_API_KEY == "sx-123"
    assert saved[0]["secrets"]["SONIOX_API_KEY"] == "sx-123"
    dlg.deleteLater()


def test_error_source_tags():
    from interview_copilot.gui.app import error_source

    assert error_source("Audio device lost after retries: x") == "audio"
    assert error_source("OpenRouter API key not set. Add it.") == "openrouter"
    assert error_source("No candidate profile found.") == "profile"
    assert error_source("Processing error: boom") is None
