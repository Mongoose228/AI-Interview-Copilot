"""Hand-off between the capture thread (producer) and the STT thread (consumer)."""

from __future__ import annotations

import queue
import threading

from ..logging_config import logger
from ..models import SpeechPhrase

_FINAL_QUEUE_SIZE = 20


class PhraseQueues:
    """
    Final phrases go to a bounded FIFO and are never dropped; partial snapshots
    go to a single-slot mailbox that keeps only the newest one. Finals always
    win: queuing a final evicts its own stale partial, and the STT worker
    drains finals before looking at the mailbox.
    """

    def __init__(self, stop_event: threading.Event):
        self.finals: queue.Queue = queue.Queue(maxsize=_FINAL_QUEUE_SIZE)
        self.partials: queue.Queue = queue.Queue(maxsize=1)
        # Set while the STT engine is running; capture skips new partials meanwhile.
        self.stt_busy = threading.Event()
        self._stop_event = stop_event
        self._lock = threading.Lock()
        self._final_ids: set = set()

    @property
    def final_ids(self) -> set:
        """Ids of finals queued or in progress (their partials are obsolete)."""
        with self._lock:
            return set(self._final_ids)

    def put_partial(self, phrase: SpeechPhrase) -> None:
        """Keep only the newest partial snapshot."""
        try:
            self.partials.put_nowait(phrase)
        except queue.Full:
            try:
                self.partials.get_nowait()
            except queue.Empty:
                pass
            try:
                self.partials.put_nowait(phrase)
            except queue.Full:
                pass

    def put_final(self, phrase: SpeechPhrase) -> None:
        """Queue a final phrase, blocking (until stop) rather than dropping it."""
        with self._lock:
            self._final_ids.add(phrase.id)
        try:
            queued_partial = self.partials.get_nowait()
            if queued_partial.id != phrase.id:
                self.put_partial(queued_partial)
        except queue.Empty:
            pass
        while not self._stop_event.is_set():
            try:
                self.finals.put(phrase, timeout=0.2)
                return
            except queue.Full:
                logger.warning("Final phrase queue full; waiting for STT worker.")

    def put_sentinel(self) -> None:
        """Wake the STT worker so it notices shutdown."""
        try:
            self.finals.put_nowait(None)
        except queue.Full:
            pass

    def get(self, timeout: float) -> SpeechPhrase | None:
        """Next phrase, finals first. ``None`` is the shutdown sentinel.

        Raises ``queue.Empty`` when nothing arrives within ``timeout``.
        """
        try:
            return self.finals.get_nowait()
        except queue.Empty:
            return self.partials.get(timeout=timeout)

    def wants_partial(self) -> bool:
        """A new partial is useful only when the STT worker is idle."""
        return not self.stt_busy.is_set() and self.finals.empty()

    def is_superseded(self, phrase: SpeechPhrase) -> bool:
        """True for a partial whose final is already queued or being transcribed."""
        if phrase.is_final:
            return False
        with self._lock:
            return phrase.id in self._final_ids

    def mark_done(self, phrase: SpeechPhrase) -> None:
        if phrase.is_final:
            with self._lock:
                self._final_ids.discard(phrase.id)

    def clear(self) -> None:
        for q in (self.finals, self.partials):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        with self._lock:
            self._final_ids.clear()
