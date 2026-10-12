"""STT thread: queued VAD phrases → transcripts on the orchestrator's queue."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from ..logging_config import logger
from ..models import SpeechPhrase
from ..stt.base import SttError
from ..stt.dual_path import DualPathSttEngine

if TYPE_CHECKING:
    from .core import InterviewPipeline


class SttWorker:
    """Single consumer of ``PhraseQueues``: one in-flight inference at a time."""

    def __init__(self, pipeline: InterviewPipeline):
        self._p = pipeline

    def run(self) -> None:
        p = self._p
        logger.info("STT thread started.")
        while not p._stop_event.is_set():
            try:
                phrase = p.queues.get(timeout=0.2)
            except queue.Empty:
                continue
            if phrase is None:
                break
            if not hasattr(phrase, "audio_data"):
                continue
            try:
                self.transcribe(phrase)
            except SttError as e:
                logger.error(f"STT worker error: {e}")
                p._emit_error(f"Transcription error: {e}", transient=True)
            except Exception as e:
                logger.exception(f"STT worker error: {e}")
                p._emit_error(f"Transcription error: {e}", transient=True)
            finally:
                p.queues.mark_done(phrase)
        logger.info("STT thread stopped.")

    def transcribe(self, phrase: SpeechPhrase) -> None:
        p = self._p
        if p.queues.is_superseded(phrase):
            return
        p._emit_status("Transcribing…")
        p.queues.stt_busy.set()
        try:
            if phrase.is_final:
                transcript = p.stt.transcribe(phrase)
            elif isinstance(p.stt, DualPathSttEngine):
                # Partials are UI-only: use the fast provisional model, never the final engine.
                transcript = p.stt.maybe_provisional(phrase)
            else:
                transcript = p.stt.transcribe(phrase)
        finally:
            p.queues.stt_busy.clear()

        if not p._publish_transcript(transcript) and not p._pause_event.is_set():
            p._emit_status("Listening…")
