"""Async orchestration: transcripts → cards, debounce/merge → one LLM task at a time."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

from ..config import config
from ..logging_config import logger
from ..models import PipelineResult, SuggestionStatus, Transcript
from ..suggestion.phrase_filter import is_filler
from .processing import process_transcript

if TYPE_CHECKING:
    from .core import InterviewPipeline

_MAX_HISTORY = 100
_HISTORY_FOR_LLM = 5


class TranscriptOrchestrator:
    """
    Consumes ``pipeline.transcript_queue`` on the event loop.

    One question can span several STT segments: partials and finals for all of
    them upsert a single card (``active_card_id``). Finals accumulate in
    ``pending`` until the debounce fires, then are merged and sent to the LLM.
    A newer question cancels the in-flight LLM task.
    """

    def __init__(self, pipeline: InterviewPipeline):
        self._p = pipeline
        self.pending: list[Transcript] = []
        self.active_card_id: uuid.UUID | None = None
        self._llm_task: asyncio.Task | None = None
        self._debounce_task: asyncio.Task | None = None
        self._latest_partial_revisions: dict = {}
        self._first_partial_cards: set = set()
        # STT phrase id -> card id it was displayed on (for cleanup on empty finals).
        self._phrase_card_ids: dict = {}

    async def run(self) -> None:
        p = self._p
        logger.info("Async orchestrator started.")
        self._report_startup_problems()
        await p.suggester.warm_up()

        while not p._stop_event.is_set():
            try:
                transcript = await asyncio.wait_for(p.transcript_queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Orchestrator queue error: {e}")
                continue

            # Dual-channel: never send candidate mic speech to the LLM path.
            if getattr(transcript, "speaker", "interviewer") == "candidate":
                continue
            try:
                self._handle(transcript)
            except Exception as e:
                logger.error(f"Orchestrator processing error: {e}")
                p._emit_error(f"Processing error: {e}", transient=True)

        for task in list(p._background_tasks):
            task.cancel()
        logger.info("Async orchestrator stopped.")

    def _report_startup_problems(self) -> None:
        p = self._p
        with p._profile_lock:
            p._active_profile = p.profile_mgr.load_active_profile()
            active_profile = p._active_profile

        if not p.suggester.is_configured:
            p._suggestions_disabled_reason = "no_key"
            p._emit_error("OpenRouter API key not set. Add it in Settings or .env.")
        elif not active_profile:
            p._suggestions_disabled_reason = "no_profile"
            p._emit_error(
                "No candidate profile found. Create a .md file in the context directory."
            )

        if config.TRANSLATION_BACKEND == "deepl":
            if not p.translator or not getattr(p.translator, "is_available", False):
                p._emit_error(
                    "DeepL translation unavailable. "
                    "Set DEEPL_API_KEY or use TRANSLATION_BACKEND=none."
                )
        elif (
            config.TRANSLATION_BACKEND == "nllb"
            and p.translator
            and not p.translator.is_available
        ):
            p._emit_error(
                "NLLB translation unavailable. Install with: pip install -e '.[nllb]'"
            )

    # --- transcript handling (event loop) --------------------------------------

    def _handle(self, transcript: Transcript) -> None:
        has_text = bool(transcript.text_en and transcript.text_en.strip())
        if not has_text:
            if transcript.is_final:
                self._on_empty_final(transcript)
            return
        if transcript.is_final:
            self._on_final(transcript)
        else:
            self._on_partial(transcript)

    def _on_empty_final(self, transcript: Transcript) -> None:
        """STT heard nothing: drop the provisional card unless it holds earlier segments."""
        card_id = self._phrase_card_ids.pop(transcript.phrase_id, None)
        self._latest_partial_revisions.pop(transcript.phrase_id, None)
        if not self.pending and card_id is not None:
            self._remove_card(card_id)

    def _on_partial(self, transcript: Transcript) -> None:
        previous = self._latest_partial_revisions.get(transcript.phrase_id, -1)
        if transcript.revision <= previous:
            return  # out-of-order snapshot
        self._latest_partial_revisions[transcript.phrase_id] = transcript.revision

        if self.active_card_id is None:
            self.active_card_id = transcript.phrase_id
        card_id = self.active_card_id
        self._phrase_card_ids[transcript.phrase_id] = card_id

        prefix = " ".join(t.text_en for t in self.pending)
        display = " ".join(part for part in (prefix, transcript.text_en) if part)
        self._show_card(card_id, display, provisional=True)

        if card_id not in self._first_partial_cards:
            self._first_partial_cards.add(card_id)
            if transcript.speech_started_at:
                latency = max(0.0, time.time() - transcript.speech_started_at)
                logger.info(f"[Latency] speech_start_to_first_partial_visible={latency:.3f}s")

    def _on_final(self, transcript: Transcript) -> None:
        self._latest_partial_revisions.pop(transcript.phrase_id, None)
        self._phrase_card_ids.pop(transcript.phrase_id, None)

        if is_filler(transcript.text_en):
            logger.debug(f"Dropped filler from LLM processing: '{transcript.text_en}'")
            if not self.pending and self.active_card_id is not None:
                self._remove_card(self.active_card_id)
            return

        if self.active_card_id is None:
            self.active_card_id = transcript.phrase_id
        self.pending.append(transcript)

        visible_at = self._show_card(
            self.active_card_id,
            " ".join(t.text_en for t in self.pending),
            provisional=False,
        )
        if visible_at and transcript.speech_ended_at:
            logger.info(
                "[Latency] "
                "speech_end_to_final_visible="
                f"{max(0.0, visible_at - transcript.speech_ended_at):.3f}s "
                "final_stt_to_visible="
                f"{max(0.0, visible_at - transcript.stt_ended_at):.3f}s"
            )
        self._restart_debounce()

    # --- cards -------------------------------------------------------------------

    def _show_card(self, card_id, text: str, *, provisional: bool) -> float | None:
        """Upsert the card; returns the time it was shown (None without a GUI)."""
        p = self._p
        if not p._transcript_callback:
            return None
        with p._profile_lock:
            profile = p._active_profile
        shown_at = time.time()
        p._transcript_callback(
            PipelineResult(
                id=card_id,
                transcript=text,
                translation_ru=None,
                suggestion=None,
                profile=profile,
                created_at=shown_at,
                suggestion_status=SuggestionStatus.PENDING,
                is_provisional=provisional,
            )
        )
        return shown_at

    def _remove_card(self, card_id) -> None:
        if self._p._transcript_remove_callback:
            self._p._transcript_remove_callback(card_id)
        self._first_partial_cards.discard(card_id)
        if self.active_card_id == card_id:
            self.active_card_id = None

    # --- debounce → LLM ----------------------------------------------------------

    def _restart_debounce(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        task = asyncio.create_task(self._flush_pending())
        self._debounce_task = task
        self._p._background_tasks.add(task)
        task.add_done_callback(self._p._background_tasks.discard)

    async def _flush_pending(self) -> None:
        p = self._p
        await asyncio.sleep(config.LLM_DEBOUNCE_DELAY_S)
        # A new phrase may already be in progress before its first partial STT result.
        # Keep the current question open until VAD finalizes that phrase (bounded).
        deadline = time.monotonic() + config.LLM_MAX_WAIT_FOR_SPEECH_END_S
        while getattr(p.vad, "is_speaking", False) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        if not self.pending:
            return

        segments = self.pending
        self.pending = []
        card_id = self.active_card_id or segments[0].phrase_id
        self.active_card_id = None
        self._first_partial_cards.discard(card_id)

        merged = merge_segments(segments, card_id)
        logger.info(
            f"Debounce: merged {len(segments)} segment(s) into one "
            f"({len(merged.text_en)} chars)"
        )

        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()

        p.transcript_history.append(merged)
        if len(p.transcript_history) > _MAX_HISTORY:
            p.transcript_history = p.transcript_history[-_MAX_HISTORY:]
        history_snapshot = list(p.transcript_history[-_HISTORY_FOR_LLM:])

        with p._profile_lock:
            profile = p._active_profile

        # Final upsert keeps the same card used by partial/final STT.
        self._show_card(card_id, merged.text_en, provisional=False)

        task = asyncio.create_task(process_transcript(p, merged, profile, history_snapshot))
        self._llm_task = task
        p._background_tasks.add(task)

        def on_task_done(t: asyncio.Task) -> None:
            p._background_tasks.discard(t)
            p._tokens_seen.discard(card_id)
            try:
                exc = t.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    logger.error(f"Task failed with error: {exc}")
                    p._emit_error(f"Task error: {exc}", transient=True)
            except asyncio.CancelledError:
                pass

        task.add_done_callback(on_task_done)


def merge_segments(segments: list[Transcript], card_id) -> Transcript:
    """Combine consecutive final segments of one question into a single transcript."""
    return Transcript(
        phrase_id=card_id,
        text_en=" ".join(t.text_en for t in segments),
        language=segments[0].language,
        confidence=min(t.confidence for t in segments),
        stt_duration_s=sum(t.stt_duration_s for t in segments),
        stt_started_at=min(
            (t.stt_started_at for t in segments if t.stt_started_at), default=0.0
        ),
        stt_ended_at=max((t.stt_ended_at for t in segments if t.stt_ended_at), default=0.0),
        is_final=True,
        revision=max(t.revision for t in segments),
        speech_started_at=min(
            (t.speech_started_at for t in segments if t.speech_started_at), default=0.0
        ),
        speech_ended_at=max(
            (t.speech_ended_at for t in segments if t.speech_ended_at), default=0.0
        ),
        audio_duration_s=sum(t.audio_duration_s for t in segments),
    )
