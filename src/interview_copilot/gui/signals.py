from PySide6.QtCore import QObject, Signal

from ..models import PipelineResult


class PipelineSignals(QObject):
    # Signal emitted when a new transcript is ready (Phase 1 of render)
    transcript_ready = Signal(object)
    # Signal emitted when a suggestion is ready (Phase 2 of render)
    suggestion_ready = Signal(object)
    suggestion_token = Signal(object, str)
    # Signal emitted when a non-fatal pipeline error occurs
    error_occurred = Signal(str)
    # Signal emitted for general status updates (e.g. loading models)
    status_changed = Signal(str)
    # Signal emitted for audio status updates (is_connected, message)
    audio_status_changed = Signal(bool, str)

