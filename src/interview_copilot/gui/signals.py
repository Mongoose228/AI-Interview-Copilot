from PySide6.QtCore import QObject, Signal


class PipelineSignals(QObject):
    transcript_ready = Signal(object)
    suggestion_ready = Signal(object)
    suggestion_token = Signal(object, str)
    # msg, transient
    error_occurred = Signal(str, bool)
    status_changed = Signal(str)
    audio_status_changed = Signal(bool, str)
