import asyncio
import queue
import threading
import time
from typing import Literal

from .audio.soundcard_wasapi import SoundCardWASAPIBackend
from .config import config
from .logging_config import log_sensitive, logger
from .models import PipelineResult, StageTiming, SuggestionStatus, Transcript
from .stt.whisper_engine import WhisperEngine
from .suggestion.openrouter import OpenRouterSuggester
from .suggestion.phrase_filter import is_filler
from .suggestion.profile_manager import ProfileManager
from .translation.deepl import DeepLTranslator
from .translation.nllb import NLLBTranslator
from .vad.silero import SileroVAD

_MAX_HISTORY = 100
_MAX_AUDIO_RETRIES = 5
_AUDIO_RETRY_BASE_DELAY = 1.0


class InterviewPipeline:
    def __init__(
        self,
        audio=None,
        vad=None,
        stt=None,
        translator=...,
        suggester=None,
        profile_mgr=None,
    ):
        self.audio = audio or SoundCardWASAPIBackend()
        self.vad = vad or SileroVAD()
        self._vad_init_error = not self.vad.is_available
        if self._vad_init_error:
            logger.error("VAD model failed to load. Speech detection will not work.")

        self.stt = stt or WhisperEngine()

        if translator is ...:
            self.translator = None
            if config.TRANSLATION_BACKEND == "nllb":
                self.translator = NLLBTranslator()
                if not self.translator.is_available:
                    logger.error(
                        "NLLB translator failed to initialize. "
                        "Install with: pip install -e '.[nllb]'"
                    )
            elif config.TRANSLATION_BACKEND == "deepl":
                self.translator = DeepLTranslator()
        else:
            self.translator = translator

        self.suggester = suggester or OpenRouterSuggester()
        self.profile_mgr = profile_mgr or ProfileManager()

        self.phrase_queue = queue.Queue(maxsize=20)
        self.transcript_queue = None
        self._loop = None

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = paused
        self._threads = []
        self._device_id: str | None = None
        self._restart_capture = threading.Event()

        self._transcript_callback = None
        self._suggestion_callback = None
        self._token_callback = None
        self._audio_status_callback = None
        self._error_callback = None
        self._status_callback = None

        self.transcript_history: list[Transcript] = []
        self._active_profile = None
        self._profile_lock = threading.Lock()
        self._suggestions_disabled_reason: Literal[
            "none", "no_key", "no_profile"
        ] = "none"
        self._tokens_seen: set = set()  # phrase ids that received stream tokens

    def set_callbacks(
        self,
        transcript_callback,
        suggestion_callback,
        token_callback,
        error_callback,
        audio_status_callback,
        status_callback=None,
    ):
        self._transcript_callback = transcript_callback
        self._suggestion_callback = suggestion_callback
        self._token_callback = token_callback
        self._error_callback = error_callback
        self._audio_status_callback = audio_status_callback
        self._status_callback = status_callback

    def set_active_profile(self, name: str) -> bool:
        snapshot = self.profile_mgr.load_profile(name)
        if not snapshot:
            return False
        with self._profile_lock:
            self._active_profile = snapshot
            if self._suggestions_disabled_reason == "no_profile":
                if self.suggester.is_configured:
                    self._suggestions_disabled_reason = "none"
        return True

    def clear_history(self):
        self.transcript_history.clear()

    def pause_capture(self):
        self._pause_event.set()
        try:
            self.audio.stop()
        except Exception as e:
            logger.warning(f"Error pausing audio: {e}")
        if self._status_callback:
            self._status_callback("Paused")

    def resume_capture(self):
        self._pause_event.clear()
        self._restart_capture.set()
        if self._status_callback:
            self._status_callback("Listening…")

    def change_device(self, device_id: str | None):
        self._device_id = device_id
        try:
            self.audio.stop()
        except Exception as e:
            logger.warning(f"Error stopping audio for device change: {e}")
        self._restart_capture.set()

    def _capture_and_vad_worker(self, device_id: str | None = None):
        logger.info("Audio capture thread started.")
        if self._vad_init_error:
            logger.error("VAD unavailable, capture thread cannot produce results.")
            if self._error_callback:
                self._error_callback(
                    "VAD model failed to load. "
                    "Speech detection is disabled. Check logs for details."
                )
            return
        retries = 0
        self._device_id = device_id

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self._stop_event.wait(0.2)
                continue

            try:
                current_device = self._device_id
                self.audio.start(current_device)
                self.vad.reset()
                retries = 0
                self._restart_capture.clear()

                if self._audio_status_callback:
                    self._audio_status_callback(True, "Audio capture active")
                if self._status_callback:
                    self._status_callback("Listening…")

                while not self._stop_event.is_set():
                    if self._pause_event.is_set() or self._restart_capture.is_set():
                        break

                    try:
                        chunk = self.audio.read_chunk()
                    except RuntimeError:
                        if self._stop_event.is_set() or self._pause_event.is_set():
                            break
                        raise

                    phrases = self.vad.process_chunk(chunk)
                    for p in phrases:
                        try:
                            self.phrase_queue.put_nowait(p)
                        except queue.Full:
                            try:
                                self.phrase_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.phrase_queue.put_nowait(p)
                            except queue.Full:
                                pass
                            logger.warning("Phrase queue full, dropped oldest phrase.")

            except (RuntimeError, ValueError, TypeError, OSError) as e:
                if self._stop_event.is_set() or self._pause_event.is_set():
                    break
                if self._restart_capture.is_set():
                    self._restart_capture.clear()
                    try:
                        self.audio.stop()
                    except Exception:
                        pass
                    continue

                logger.error(f"Capture worker error: {e}")
                try:
                    self.audio.stop()
                except (RuntimeError, ValueError, TypeError) as stop_err:
                    logger.warning(f"Error ignored: {stop_err}")

                retries += 1
                if retries > _MAX_AUDIO_RETRIES:
                    logger.error(
                        f"Audio capture failed after {_MAX_AUDIO_RETRIES} retries. Giving up."
                    )
                    if self._audio_status_callback:
                        self._audio_status_callback(False, f"Audio device lost: {e}")
                    if self._error_callback:
                        self._error_callback(f"Audio device lost after retries: {e}")
                    break

                delay = _AUDIO_RETRY_BASE_DELAY * (2 ** (retries - 1))
                logger.warning(
                    f"Audio capture failed, retrying in {delay:.1f}s "
                    f"(attempt {retries}/{_MAX_AUDIO_RETRIES})"
                )
                if self._audio_status_callback:
                    self._audio_status_callback(
                        False, f"Reconnecting... ({retries}/{_MAX_AUDIO_RETRIES})"
                    )
                self._stop_event.wait(delay)

            finally:
                try:
                    self.audio.stop()
                except (RuntimeError, ValueError, TypeError) as stop_err:
                    logger.warning(f"Error ignored: {stop_err}")

        logger.info("Audio capture thread stopped.")

    def _stt_worker(self):
        logger.info("STT thread started.")
        while not self._stop_event.is_set():
            try:
                phrase = self.phrase_queue.get(timeout=0.5)
                if phrase is None:
                    break
                if not hasattr(phrase, "audio_data"):
                    continue
                if self._status_callback:
                    self._status_callback("Transcribing…")
                transcript = self.stt.transcribe(phrase)
                if transcript.text_en and self._loop and self.transcript_queue:
                    self._loop.call_soon_threadsafe(
                        self.transcript_queue.put_nowait, transcript
                    )
                elif self._status_callback and not self._pause_event.is_set():
                    self._status_callback("Listening…")
            except queue.Empty:
                continue
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logger.error(f"STT worker error: {e}")
                if self._error_callback:
                    self._error_callback(f"Transcription error: {e}", transient=True)
        logger.info("STT thread stopped.")

    def _emit_error(self, msg: str, *, transient: bool = False):
        if self._error_callback:
            try:
                self._error_callback(msg, transient=transient)
            except TypeError:
                self._error_callback(msg)

    async def _process_transcript(
        self,
        transcript: Transcript,
        active_profile,
        history_snapshot: list[Transcript],
    ):
        timings = [
            StageTiming(
                stage_name="STT",
                started_at=transcript.stt_started_at,
                ended_at=transcript.stt_ended_at or transcript.stt_duration_s,
                duration_s=transcript.stt_duration_s,
            )
        ]

        translation_coro = None
        suggestion_coro = None
        tokens_received = False

        async def timed_translate():
            start_t = time.time()
            res = await asyncio.to_thread(self.translator.translate, transcript.text_en)
            duration = time.time() - start_t
            timings.append(
                StageTiming(
                    stage_name="Translation",
                    started_at=start_t,
                    ended_at=time.time(),
                    duration_s=duration,
                )
            )
            return res

        async def stream_cb(token: str):
            nonlocal tokens_received
            tokens_received = True
            self._tokens_seen.add(transcript.phrase_id)
            if self._token_callback:
                self._token_callback(transcript.phrase_id, token)

        async def timed_suggestion():
            start_t = time.time()
            if self._status_callback:
                self._status_callback("Thinking…")
            res = await self.suggester.get_suggestion(
                history_snapshot, active_profile, stream_callback=stream_cb
            )
            duration = time.time() - start_t
            timings.append(
                StageTiming(
                    stage_name="AI_Suggestion",
                    started_at=start_t,
                    ended_at=time.time(),
                    duration_s=duration,
                )
            )
            return res

        if self.translator and getattr(self.translator, "is_available", True):
            translation_coro = timed_translate()

        suggestion_status = SuggestionStatus.PENDING
        if self._suggestions_disabled_reason == "no_key":
            suggestion_status = SuggestionStatus.SKIPPED_NO_KEY
        elif self._suggestions_disabled_reason == "no_profile" or not active_profile:
            suggestion_status = SuggestionStatus.SKIPPED_NO_PROFILE
        elif active_profile and self.suggester.is_configured:
            suggestion_coro = timed_suggestion()

        translation_ru = None
        suggestion = None

        try:
            if translation_coro and suggestion_coro:
                translation_ru, suggestion = await asyncio.gather(
                    translation_coro, suggestion_coro
                )
            elif translation_coro:
                translation_ru = await translation_coro
            elif suggestion_coro:
                suggestion = await suggestion_coro
        except asyncio.CancelledError:
            status = (
                SuggestionStatus.INTERRUPTED
                if tokens_received or transcript.phrase_id in self._tokens_seen
                else SuggestionStatus.CANCELLED
            )
            result = PipelineResult(
                id=transcript.phrase_id,
                transcript=transcript.text_en,
                translation_ru=None,
                suggestion=None,
                profile=active_profile,
                created_at=time.time(),
                is_cancelled=True,
                suggestion_status=status,
            )
            if self._suggestion_callback:
                self._suggestion_callback(result)
            raise
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self._emit_error(f"Processing error: {e}", transient=True)
            suggestion_status = SuggestionStatus.FAILED

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

        if suggestion and suggestion.answer_en and self.translator and getattr(
            self.translator, "is_available", True
        ):
            try:
                answer_ru = await asyncio.to_thread(
                    self.translator.translate,
                    suggestion.answer_en,
                )
                result = PipelineResult(
                    id=result.id,
                    transcript=result.transcript,
                    translation_ru=result.translation_ru,
                    suggestion=result.suggestion,
                    profile=result.profile,
                    timings=result.timings,
                    created_at=result.created_at,
                    answer_ru=answer_ru,
                    suggestion_status=result.suggestion_status,
                )
            except Exception as e:
                logger.warning(f"Failed to translate LLM answer: {e}")

        self._display_result(result)
        if self._suggestion_callback:
            self._suggestion_callback(result)
        if self._status_callback and not self._pause_event.is_set():
            self._status_callback("Listening…")

    async def _async_orchestrator(self):
        logger.info("Async orchestrator started.")
        with self._profile_lock:
            self._active_profile = self.profile_mgr.load_active_profile()
            active_profile = self._active_profile

        if not self.suggester.is_configured:
            self._suggestions_disabled_reason = "no_key"
            self._emit_error(
                "OpenRouter API key not set. Add OPENROUTER_API_KEY to .env.",
            )
        elif not active_profile:
            self._suggestions_disabled_reason = "no_profile"
            self._emit_error(
                "No candidate profile found. Create a .md file in the context directory.",
            )

        await self.suggester.warm_up()

        self._background_tasks = set()
        current_llm_task = None
        pending_transcripts: list[Transcript] = []
        debounce_task: asyncio.Task | None = None

        if config.TRANSLATION_BACKEND == "deepl":
            if not self.translator or not getattr(self.translator, "is_available", False):
                self._emit_error(
                    "DeepL translation unavailable. "
                    "Set DEEPL_API_KEY or use TRANSLATION_BACKEND=none."
                )
        elif (
            config.TRANSLATION_BACKEND == "nllb"
            and self.translator
            and not self.translator.is_available
        ):
            self._emit_error(
                "NLLB translation unavailable. Install with: pip install -e '.[nllb]'"
            )

        async def flush_pending():
            nonlocal current_llm_task, pending_transcripts

            await asyncio.sleep(config.LLM_DEBOUNCE_DELAY_S)

            if not pending_transcripts:
                return

            segments = pending_transcripts
            pending_transcripts = []

            merged_text = " ".join(t.text_en for t in segments)
            merged_transcript = Transcript(
                phrase_id=segments[-1].phrase_id,
                text_en=merged_text,
                language=segments[0].language,
                confidence=min(t.confidence for t in segments),
                stt_duration_s=sum(t.stt_duration_s for t in segments),
                stt_started_at=min(
                    (t.stt_started_at for t in segments if t.stt_started_at),
                    default=0.0,
                ),
                stt_ended_at=max(
                    (t.stt_ended_at for t in segments if t.stt_ended_at),
                    default=0.0,
                ),
            )

            logger.info(
                f"Debounce: merged {len(segments)} segment(s) into one "
                f"({len(merged_text)} chars)"
            )

            if current_llm_task and not current_llm_task.done():
                current_llm_task.cancel()

            self.transcript_history.append(merged_transcript)
            if len(self.transcript_history) > _MAX_HISTORY:
                self.transcript_history = self.transcript_history[-_MAX_HISTORY:]

            history_snapshot = list(self.transcript_history[-5:])

            with self._profile_lock:
                profile = self._active_profile

            # Emit card once after merge
            if self._transcript_callback:
                initial_result = PipelineResult(
                    id=merged_transcript.phrase_id,
                    transcript=merged_transcript.text_en,
                    translation_ru=None,
                    suggestion=None,
                    profile=profile,
                    created_at=time.time(),
                    suggestion_status=SuggestionStatus.PENDING,
                )
                self._transcript_callback(initial_result)

            current_llm_task = asyncio.create_task(
                self._process_transcript(merged_transcript, profile, history_snapshot)
            )
            self._background_tasks.add(current_llm_task)

            def on_task_done(t):
                self._background_tasks.discard(t)
                try:
                    exc = t.exception()
                    if exc and not isinstance(exc, asyncio.CancelledError):
                        logger.error(f"Task failed with error: {exc}")
                        self._emit_error(f"Task error: {exc}", transient=True)
                except asyncio.CancelledError:
                    pass

            current_llm_task.add_done_callback(on_task_done)

        while not self._stop_event.is_set():
            try:
                transcript = await asyncio.wait_for(
                    self.transcript_queue.get(), timeout=1.0
                )
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Orchestrator queue error: {e}")
                continue

            try:
                if not transcript.text_en or not transcript.text_en.strip():
                    continue

                if is_filler(transcript.text_en):
                    logger.debug(
                        f"Dropped filler from LLM processing: '{transcript.text_en}'"
                    )
                    continue

                pending_transcripts.append(transcript)

                if debounce_task and not debounce_task.done():
                    debounce_task.cancel()

                debounce_task = asyncio.create_task(flush_pending())
                self._background_tasks.add(debounce_task)
                debounce_task.add_done_callback(
                    lambda t: self._background_tasks.discard(t)
                )
            except Exception as e:
                logger.error(f"Orchestrator processing error: {e}")
                self._emit_error(f"Processing error: {e}", transient=True)
                continue

        for task in list(self._background_tasks):
            task.cancel()

        logger.info("Async orchestrator stopped.")

    def _display_result(self, result: PipelineResult):
        if result.timings:
            timing_parts = [f"{t.stage_name}={t.duration_s:.2f}s" for t in result.timings]
            total = sum(t.duration_s for t in result.timings)
            logger.info(f"[Timings] {' → '.join(timing_parts)} | total={total:.2f}s")

        if config.LOG_OBFUSCATION_ENABLED:
            logger.info(
                f"[Result] id={result.id} "
                f"has_translation={result.translation_ru is not None} "
                f"has_suggestion={result.suggestion is not None} "
                f"status={result.suggestion_status.value}"
            )
            return

        if self._suggestion_callback:
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

    async def start(self, device_id: str | None = None):
        self._stop_event.clear()
        self._pause_event.clear()
        self._loop = asyncio.get_running_loop()
        self.transcript_queue = asyncio.Queue()

        t_cap = threading.Thread(
            target=self._capture_and_vad_worker, args=(device_id,), daemon=True
        )
        t_stt = threading.Thread(target=self._stt_worker, daemon=True)

        self._threads = [t_cap, t_stt]
        for t in self._threads:
            t.start()

        try:
            await self._async_orchestrator()
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    def stop(self):
        self._stop_event.set()
        self._pause_event.clear()

        # Flush any in-progress speech before tearing down audio
        try:
            flushed = self.vad.flush()
            if flushed:
                try:
                    self.phrase_queue.put_nowait(flushed)
                except queue.Full:
                    pass
        except Exception as e:
            logger.warning(f"VAD flush during stop failed: {e}")

        try:
            self.audio.stop()
        except Exception as e:
            logger.warning(f"Error stopping audio during shutdown: {e}")

        try:
            self.phrase_queue.put_nowait(None)
        except queue.Full:
            pass

        for t in self._threads:
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning(f"Thread {t.name} did not terminate in time.")

        self.vad.reset()

        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get_nowait()
            except queue.Empty:
                break
        self._threads = []
