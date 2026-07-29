from PySide6.QtCore import QObject, Signal

from ..models import PipelineResult


class PipelineSignals(QObject):
    # Signal emitted when a new PipelineResult is ready
    result_ready = Signal(PipelineResult)
