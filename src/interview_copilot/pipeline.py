import asyncio
import queue
import threading
import time

from .audio.soundcard_wasapi import SoundCardWASAPIBackend
from .config import config
from .logging_config import log_sensitive, logger
from .models import PipelineResult, StageTiming, Transcript
from .stt.whisper_engine import WhisperEngine
from .suggestion.openrouter import OpenRouterSuggester
from .suggestion.profile_manager import ProfileManager
from .translation.deepl import DeepLTranslator
from .translation.nllb import NLLBTranslator
from .vad.silero import SileroVAD

# Maximum transcript history to prevent unbounded memory growth
_MAX_HISTORY = 100
# Maximum retry attempts for audio device reconnection
_MAX_AUDIO_RETRIES = 5
_AUDIO_RETRY_BASE_DELAY = 1.0  # seconds


class InterviewPipeline:
    def __init__(self):
        if config.AUDIO_BACKEND != "soundcard":
            raise NotImplementedError(
                f"Audio backend '{config.AUDIO_BACKEND}' is not implemented."
                " Use 'soundcard'."
            )
        self.audio = SoundCardWASAPIBackend()
        self.vad = SileroVAD()
        if not self.vad.is_available:
            logger.error("VAD model failed to load. Speech detection will not work.")
            self._vad_init_error = True
        else:
            self._vad_init_error = False
        self.stt = WhisperEngine()

        # Translation strategy — single config key
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

        self.suggester = OpenRouterSuggester()
        self.profile_mgr = ProfileManager()

        self.phrase_queue = queue.Queue(maxsize=20)
        self.transcript_queue = None
        self._loop = None

        self._stop_event = threading.Event()
        self._threads = []

        # Callbacks
        self._transcript_callback = None
        self._suggestion_callback = None
        self._token_callback = None
        self._audio_status_callback = None
        self._error_callback = None

        # History (bounded)
        self.transcript_history: list[Transcript] = []

    def set_callbacks(
        self,
        transcript_callback,
        suggestion_callback,
        token_callback,
        error_callback,
        audio_status_callback,
    ):
        self._transcript_callback = transcript_callback
        self._suggestion_callback = suggestion_callback
        self._token_callback = token_callback
        self._error_callback = error_callback
        self._audio_status_callback = audio_status_callback

    def _capture_and_vad_worker(self, device_id: str | None = None):
        logger.info("Audio capture thread started.")
        if getattr(self, '_vad_init_error', False):
            logger.error("VAD unavailable, capture thread cannot produce results.")
            if self._error_callback:
                self._error_callback(
                    "⚠️ VAD model failed to load. "
                    "Speech detection is disabled. Check logs for details."
                )
            return
        retries = 0

        while not self._stop_event.is_set():
            try:
                self.audio.start(device_id)
                self.vad.reset()  # Reset VAD state on (re)start
                retries = 0  # Reset retry counter on successful start

                if self._audio_status_callback:
                    self._audio_status_callback(True, "Audio capture active")

                while not self._stop_event.is_set():
                    chunk = self.audio.read_chunk()
                    phrases = self.vad.process_chunk(chunk)
                    for p in phrases:
                        try:
                            self.phrase_queue.put_nowait(p)
                        except queue.Full:
                            # Drop oldest to keep fresh phrases (real-time priority)
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
                if self._stop_event.is_set():
                    break
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
                    break  # Sentinel received, exit gracefully
                vad_to_stt_gap = time.time() - phrase.vad_end_at
                logger.debug(f"[Timing] VAD→STT gap: {vad_to_stt_gap:.3f}s")
                transcript = self.stt.transcribe(phrase)
                if transcript.text_en and self._loop and self.transcript_queue:
                    self._loop.call_soon_threadsafe(self.transcript_queue.put_nowait, transcript)
            except queue.Empty:
                continue
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logger.error(f"STT worker error: {e}")
                if self._error_callback:
                    self._error_callback(f"Transcription error: {e}")
        logger.info("STT thread stopped.")

    async def _process_transcript(
        self,
        transcript: Transcript,
        active_profile,
        history_snapshot: list[Transcript],
    ):
        """Process a single transcript: translate + get suggestion concurrently."""
        translation_coro = None
        suggestion_coro = None

        # In our refactor, we just store duration_s to avoid modifying the whole pipeline right now
        timings = [
            StageTiming(
                stage_name="STT",
                started_at=0.0,
                ended_at=transcript.stt_duration_s,
                duration_s=transcript.stt_duration_s,
            )
        ]

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

        async def timed_suggestion():
            start_t = time.time()
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

        if self.translator:
            translation_coro = timed_translate()

        async def stream_cb(token: str):
            if self._token_callback:
                self._token_callback(transcript.phrase_id, token)

        if active_profile:
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
            logger.info(f"Task for transcript '{transcript.text_en}' was cancelled.")
            result = PipelineResult(
                id=transcript.phrase_id,
                transcript=transcript.text_en,
                translation_ru=None,
                suggestion=None,
                profile=active_profile,
                created_at=time.time(),
                is_cancelled=True,
            )
            if self._suggestion_callback:
                self._suggestion_callback(result)
            raise
        except Exception as e:
            logger.error(f"Processing error: {e}")
            if self._error_callback:
                self._error_callback(f"Processing error: {e}")

        result = PipelineResult(
            id=transcript.phrase_id,
            transcript=transcript.text_en,
            translation_ru=translation_ru,
            suggestion=suggestion,
            profile=active_profile,
            timings=timings,
            created_at=time.time(),
            answer_ru=None,
        )

        # Translate the LLM answer to RU if translator is available
        if suggestion and suggestion.answer_en and self.translator:
            try:
                answer_ru = await asyncio.to_thread(
                    self.translator.translate,
                    suggestion.answer_en,
                )
                # Rebuild result with answer_ru (frozen dataclass)
                result = PipelineResult(
                    id=result.id,
                    transcript=result.transcript,
                    translation_ru=result.translation_ru,
                    suggestion=result.suggestion,
                    profile=result.profile,
                    timings=result.timings,
                    created_at=result.created_at,
                    answer_ru=answer_ru,
                )
            except Exception as e:
                logger.warning(f"Failed to translate LLM answer: {e}")

        self._display_result(result)
        if self._suggestion_callback:
            self._suggestion_callback(result)

    async def _async_orchestrator(self):
        """Main async loop: accumulate transcripts, debounce, then send to LLM."""
        logger.info("Async orchestrator started.")
        active_profile = self.profile_mgr.load_active_profile()

        # Pre-warm OpenRouter connection (DNS + TLS)
        await self.suggester.warm_up()

        self._background_tasks = set()
        current_llm_task = None
        pending_transcripts: list[Transcript] = []
        debounce_task: asyncio.Task | None = None

        # Report NLLB initialization failure to GUI (callback is now set)
        if (config.TRANSLATION_BACKEND == "nllb"
                and self.translator
                and not self.translator.is_available):
            if self._error_callback:
                self._error_callback(
                    "⚠️ NLLB translation unavailable. "
                    "Install with: pip install -e '.[nllb]'"
                )

        from .suggestion.phrase_filter import is_filler

        async def flush_pending():
            """Wait for debounce delay, then merge and send to LLM."""
            nonlocal current_llm_task, pending_transcripts

            await asyncio.sleep(config.LLM_DEBOUNCE_DELAY_S)

            if not pending_transcripts:
                return

            # Merge all accumulated transcript segments
            segments = pending_transcripts
            pending_transcripts = []

            merged_text = " ".join(t.text_en for t in segments)
            merged_transcript = Transcript(
                phrase_id=segments[-1].phrase_id,  # Use last segment's ID
                text_en=merged_text,
                language=segments[0].language,
                confidence=min(t.confidence for t in segments),
                stt_duration_s=sum(t.stt_duration_s for t in segments),
            )

            logger.info(
                f"Debounce: merged {len(segments)} segment(s) into one "
                f"({len(merged_text)} chars)"
            )

            # Cancel previous LLM task only now (not on every fragment)
            if current_llm_task and not current_llm_task.done():
                current_llm_task.cancel()

            self.transcript_history.append(merged_transcript)
            if len(self.transcript_history) > _MAX_HISTORY:
                self.transcript_history = self.transcript_history[-_MAX_HISTORY:]

            history_snapshot = list(self.transcript_history[-5:])

            current_llm_task = asyncio.create_task(
                self._process_transcript(
                    merged_transcript, active_profile, history_snapshot
                )
            )
            self._background_tasks.add(current_llm_task)

            def on_task_done(t):
                self._background_tasks.discard(t)
                try:
                    exc = t.exception()
                    if exc and not isinstance(exc, asyncio.CancelledError):
                        logger.error(f"Task failed with error: {exc}")
                        if self._error_callback:
                            self._error_callback(f"Task error: {exc}")
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
                # Early exit for filler phrases
                if is_filler(transcript.text_en):
                    logger.debug(
                        f"Dropped filler from LLM processing: '{transcript.text_en}'"
                    )
                    continue

                # Emit early transcript card to GUI immediately
                if self._transcript_callback:
                    initial_result = PipelineResult(
                        id=transcript.phrase_id,
                        transcript=transcript.text_en,
                        translation_ru=None,
                        suggestion=None,
                        profile=active_profile,
                        created_at=time.time(),
                    )
                    self._transcript_callback(initial_result)

                # Accumulate transcript segment
                pending_transcripts.append(transcript)

                # Reset debounce timer: cancel old, start new
                if debounce_task and not debounce_task.done():
                    debounce_task.cancel()

                debounce_task = asyncio.create_task(flush_pending())
                self._background_tasks.add(debounce_task)
                debounce_task.add_done_callback(
                    lambda t: self._background_tasks.discard(t)
                )
            except Exception as e:
                logger.error(f"Orchestrator processing error: {e}")
                if self._error_callback:
                    self._error_callback(f"Processing error: {e}")
                continue  # Keep processing next phrases

        # Cleanup: cancel any remaining tasks on shutdown
        for task in list(self._background_tasks):
            task.cancel()

        logger.info("Async orchestrator stopped.")

    def _display_result(self, result: PipelineResult):
        # Log stage timings for performance analysis (no private data, always log)
        if result.timings:
            timing_parts = [f"{t.stage_name}={t.duration_s:.2f}s" for t in result.timings]
            total = sum(t.duration_s for t in result.timings)
            logger.info(f"[Timings] {' → '.join(timing_parts)} | total={total:.2f}s")

        if config.LOG_OBFUSCATION_ENABLED:
            # In privacy mode, only log metadata
            logger.info(
                f"[Result] id={result.id} "
                f"has_translation={result.translation_ru is not None} "
                f"has_suggestion={result.suggestion is not None}"
            )
            return

        # GUI mode: log sensitive data via log_sensitive(), don't print to stdout
        if self._suggestion_callback:
            log_sensitive(f"[EN]: {result.transcript}")
            if result.translation_ru:
                log_sensitive(f"[RU]: {result.translation_ru}")
            if result.suggestion:
                log_sensitive(f"[AI EN]: {result.suggestion.answer_en}")
            if getattr(result, "answer_ru", None):
                log_sensitive(f"[AI RU]: {result.answer_ru}")
            return

        # CLI mode: print to console
        print("\n" + "=" * 60)
        log_sensitive(f"[EN]: {result.transcript}")
        print(f"🗣️  [EN]: {result.transcript}")
        if result.translation_ru:
            log_sensitive(f"[RU]: {result.translation_ru}")
            print(f"🇷🇺  [RU]: {result.translation_ru}")
        if result.suggestion:
            print("-" * 60)
            hedging_mark = " ⚠️(uncertain)" if result.suggestion.has_hedging else ""
            log_sensitive(f"[AI EN]: {result.suggestion.answer_en}")
            print(f"💡 [AI EN]{hedging_mark}: {result.suggestion.answer_en}")
            if getattr(result, "answer_ru", None):
                log_sensitive(f"[AI RU]: {result.answer_ru}")
                print(f"💡 [AI RU]: {result.answer_ru}")
        print("=" * 60 + "\n")

    async def start(self, device_id: str | None = None):
        self._stop_event.clear()
        self._loop = asyncio.get_running_loop()
        self.transcript_queue = asyncio.Queue()

        # Start Threads
        t_cap = threading.Thread(
            target=self._capture_and_vad_worker, args=(device_id,), daemon=True
        )
        t_stt = threading.Thread(target=self._stt_worker, daemon=True)

        self._threads = [t_cap, t_stt]
        for t in self._threads:
            t.start()

        # Run async orchestrator
        try:
            await self._async_orchestrator()
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    def stop(self):
        self._stop_event.set()

        # Force-stop audio to unblock read_chunk() in capture thread
        try:
            self.audio.stop()
        except Exception as e:
            logger.warning(f"Error stopping audio during shutdown: {e}")

        # Send sentinel to unblock STT worker waiting on phrase_queue
        try:
            self.phrase_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for threads to exit
        for t in self._threads:
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning(f"Thread {t.name} did not terminate in time.")

        self.vad.reset()  # Reset VAD state on stop

        # Drain queues for clean restart
        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get_nowait()
            except queue.Empty:
                break
        self._threads = []
