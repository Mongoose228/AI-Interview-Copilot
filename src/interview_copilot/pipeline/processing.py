"""Per-question work: translation + streamed LLM suggestion → PipelineResult."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import TYPE_CHECKING

from ..config import config
from ..logging_config import log_sensitive, logger
from ..models import PipelineResult, StageTiming, SuggestionStatus, Transcript

if TYPE_CHECKING:
    from .core import InterviewPipeline


async def _none():
    return None


async def process_transcript(
    p: InterviewPipeline,
    transcript: Transcript,
    active_profile,
    history_snapshot: list[Transcript],
) -> None:
    """Translate the question and stream an answer; reports via pipeline callbacks.

    Cancellation (a newer question arrived) reports INTERRUPTED/CANCELLED and re-raises.
    """
    timings = [
        StageTiming(
            stage_name="STT",
            started_at=transcript.stt_started_at,
            ended_at=transcript.stt_ended_at
            or transcript.stt_started_at + transcript.stt_duration_s,
            duration_s=transcript.stt_duration_s,
        )
    ]
    tokens_received = False
    first_token_logged = False
    translator = p.translator
    suggester = p.suggester
    can_translate = bool(translator) and getattr(translator, "is_available", True)

    async def timed_translate():
        start_t = time.time()
        res = await asyncio.to_thread(translator.translate, transcript.text_en)
        timings.append(
            StageTiming(
                stage_name="Translation",
                started_at=start_t,
                ended_at=time.time(),
                duration_s=time.time() - start_t,
            )
        )
        return res

    async def stream_cb(token: str):
        nonlocal tokens_received, first_token_logged
        tokens_received = True
        p._tokens_seen.add(transcript.phrase_id)
        if not first_token_logged:
            first_token_logged = True
            if transcript.speech_ended_at:
                latency = max(0.0, time.time() - transcript.speech_ended_at)
                logger.info(f"[Latency] speech_end_to_first_llm_token={latency:.3f}s")
        if p._token_callback:
            p._token_callback(transcript.phrase_id, token)

    async def timed_suggestion():
        start_t = time.time()
        p._emit_status("Thinking…")
        res = await suggester.get_suggestion(
            history_snapshot, active_profile, stream_callback=stream_cb
        )
        timings.append(
            StageTiming(
                stage_name="AI_Suggestion",
                started_at=start_t,
                ended_at=time.time(),
                duration_s=time.time() - start_t,
            )
        )
        return res

    translation_coro = timed_translate() if can_translate else None
    suggestion_coro = None
    suggestion_status = SuggestionStatus.PENDING
    if p._suggestions_disabled_reason == "no_key":
        suggestion_status = SuggestionStatus.SKIPPED_NO_KEY
    elif p._suggestions_disabled_reason == "no_profile" or not active_profile:
        suggestion_status = SuggestionStatus.SKIPPED_NO_PROFILE
    elif suggester.is_configured:
        suggestion_coro = timed_suggestion()

    try:
        translation_res, suggestion_res = await asyncio.gather(
            translation_coro or _none(),
            suggestion_coro or _none(),
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        status = (
            SuggestionStatus.INTERRUPTED
            if tokens_received or transcript.phrase_id in p._tokens_seen
            else SuggestionStatus.CANCELLED
        )
        if p._suggestion_callback:
            p._suggestion_callback(
                PipelineResult(
                    id=transcript.phrase_id,
                    transcript=transcript.text_en,
                    translation_ru=None,
                    suggestion=None,
                    profile=active_profile,
                    created_at=time.time(),
                    is_cancelled=True,
                    suggestion_status=status,
                )
            )
        raise

    translation_ru = None
    suggestion = None
    if isinstance(translation_res, BaseException):
        logger.warning(f"Translation failed: {translation_res}")
    else:
        translation_ru = translation_res
    if isinstance(suggestion_res, BaseException):
        logger.error(f"Processing error: {suggestion_res}")
        p._emit_error(f"Processing error: {suggestion_res}", transient=True)
        suggestion_status = SuggestionStatus.FAILED
    else:
        suggestion = suggestion_res

    if suggestion_coro is not None:
        if suggestion is None and suggestion_status == SuggestionStatus.PENDING:
            suggestion_status = SuggestionStatus.FAILED
        elif suggestion is not None:
            suggestion_status = SuggestionStatus.OK

    result = PipelineResult(
        id=transcript.phrase_id,
        transcript=transcript.text_en,
        translation_ru=translation_ru,
        suggestion=suggestion,
        profile=active_profile,
        timings=timings,
        created_at=time.time(),
        answer_ru=None,
        suggestion_status=suggestion_status,
    )

    if suggestion and suggestion.answer_en and can_translate:
        try:
            answer_ru = await asyncio.to_thread(translator.translate, suggestion.answer_en)
            result = dataclasses.replace(result, answer_ru=answer_ru)
        except Exception as e:
            logger.warning(f"Failed to translate LLM answer: {e}")

    display_result(result, has_gui=p._suggestion_callback is not None)
    if p._suggestion_callback:
        p._suggestion_callback(result)
    if not p._pause_event.is_set():
        p._emit_status("Listening…")


def display_result(result: PipelineResult, *, has_gui: bool) -> None:
    """Log timings and (unless obfuscated) the texts; print them in CLI mode."""
    if result.timings:
        timing_parts = [f"{t.stage_name}={t.duration_s:.2f}s" for t in result.timings]
        total = sum(t.duration_s for t in result.timings)
        logger.info(f"[Timings] {' -> '.join(timing_parts)} | total={total:.2f}s")

    if config.LOG_OBFUSCATION_ENABLED:
        logger.info(
            f"[Result] id={result.id} "
            f"has_translation={result.translation_ru is not None} "
            f"has_suggestion={result.suggestion is not None} "
            f"status={result.suggestion_status.value}"
        )
        return

    if has_gui:
        log_sensitive(f"[EN]: {result.transcript}")
        if result.translation_ru:
            log_sensitive(f"[RU]: {result.translation_ru}")
        if result.suggestion:
            log_sensitive(f"[AI EN]: {result.suggestion.answer_en}")
        if result.answer_ru:
            log_sensitive(f"[AI RU]: {result.answer_ru}")
        return

    print("\n" + "=" * 60)
    log_sensitive(f"[EN]: {result.transcript}")
    print(f"[EN]: {result.transcript}")
    if result.translation_ru:
        log_sensitive(f"[RU]: {result.translation_ru}")
        print(f"[RU]: {result.translation_ru}")
    if result.suggestion:
        print("-" * 60)
        hedging_mark = " (uncertain)" if result.suggestion.has_hedging else ""
        log_sensitive(f"[AI EN]: {result.suggestion.answer_en}")
        print(f"[AI EN]{hedging_mark}: {result.suggestion.answer_en}")
        if result.answer_ru:
            log_sensitive(f"[AI RU]: {result.answer_ru}")
            print(f"[AI RU]: {result.answer_ru}")
    print("=" * 60 + "\n")
