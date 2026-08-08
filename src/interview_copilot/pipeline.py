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
            raise NotImplementedError(f"Audio backend '{config.AUDIO_BACKEND}' is not implemented. Use 'soundcard'.")
        self.audio = SoundCardWASAPIBackend()
        self.vad = SileroVAD()
        self.stt = WhisperEngine()

        # Translation strategy — single config key
        self.translator = None
        if config.TRANSLATION_BACKEND == "deepl":
            self.translator = DeepLTranslator()
        elif config.TRANSLATION_BACKEND == "nllb":
            self.translator = NLLBTranslator()

        self.suggester = OpenRouterSuggester()
        self.profile_mgr = ProfileManager()

        self.phrase_queue = queue.Queue(maxsize=20)
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

    def set_callbacks(self, transcript_callback, suggestion_callback, token_callback, error_callback, audio_status_callback):
        self._transcript_callback = transcript_callback
        self._suggestion_callback = suggestion_callback
        self._token_callback = token_callback
        self._error_callback = error_callback
        self._audio_status_callback = audio_status_callback

    def _capture_and_vad_worker(self, device_id: str | None = None):
        logger.info("Audio capture thread started.")
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
                            self.phrase_queue.put(p, timeout=1.0)
                        except queue.Full:
                            logger.warning("Phrase queue is full, dropping phrase.")

            except (RuntimeError, ValueError, TypeError, OSError) as e:
                if self._stop_event.is_set():
                    break
                logger.error(f"Capture worker error: {e}")
                try:
                    self.audio.stop()
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning(f"Error ignored: {e}")

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
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning(f"Error ignored: {e}")

        logger.info("Audio capture thread stopped.")

    def _stt_worker(self):
        logger.info("STT thread started.")
        while not self._stop_event.is_set():
            try:
                phrase = self.phrase_queue.get(timeout=0.5)
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

    async def _process_transcript(self, transcript: Transcript, active_profile, history_snapshot: list[Transcript]):
        """Process a single transcript: translate + get suggestion concurrently."""
        translation_coro = None
        suggestion_coro = None
        
        # In our refactor, we just store duration_s to avoid modifying the whole pipeline right now
        timings = [StageTiming(stage_name="STT", started_at=0.0, ended_at=transcript.stt_duration_s, duration_s=transcript.stt_duration_s)]

        async def timed_translate():
            start_t = time.time()
            res = await asyncio.to_thread(self.translator.translate, transcript.text_en)
            duration = time.time() - start_t
            timings.append(StageTiming(stage_name="Translation", started_at=start_t, ended_at=time.time(), duration_s=duration))
            return res

        async def timed_suggestion():
            start_t = time.time()
            res = await self.suggester.get_suggestion(
                history_snapshot, active_profile, stream_callback=stream_cb
            )
            duration = time.time() - start_t
            timings.append(StageTiming(stage_name="AI_Suggestion", started_at=start_t, ended_at=time.time(), duration_s=duration))
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
            raise
        except (RuntimeError, ValueError, TypeError, OSError) as e:
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
        )

        self._display_result(result)
        if self._suggestion_callback:
            self._suggestion_callback(result)

    async def _async_orchestrator(self):
        logger.info("Async orchestrator started.")
        active_profile = self.profile_mgr.load_active_profile()

        self._background_tasks = set()
        current_llm_task = None

        from .suggestion.phrase_filter import is_filler

        while not self._stop_event.is_set():
            try:
                transcript = await asyncio.wait_for(self.transcript_queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            # Emit early transcript to GUI
            if self._transcript_callback:
                initial_result = PipelineResult(
                    id=transcript.phrase_id,
                    transcript=transcript.text_en,
                    translation_ru=None,
                    suggestion=None,
                    profile=active_profile,
                    created_at=time.time()
                )
                self._transcript_callback(initial_result)

            self.transcript_history.append(transcript)

            # Trim history to prevent unbounded growth
            if len(self.transcript_history) > _MAX_HISTORY:
                self.transcript_history = self.transcript_history[-_MAX_HISTORY:]

            # Cancel-and-resend logic
            if is_filler(transcript.text_en):
                logger.debug(f"Dropped filler from LLM processing: '{transcript.text_en}'")
                continue

            if current_llm_task and not current_llm_task.done():
                current_llm_task.cancel()

            history_snapshot = list(self.transcript_history[-5:])

            current_llm_task = asyncio.create_task(
                self._process_transcript(transcript, active_profile, history_snapshot)
            )
            self._background_tasks.add(current_llm_task)
            current_llm_task.add_done_callback(self._background_tasks.discard)

        # Cleanup: cancel any remaining tasks on shutdown
        for task in list(self._background_tasks):
            task.cancel()

        logger.info("Async orchestrator stopped.")

    def _display_result(self, result: PipelineResult):
        if config.LOG_OBFUSCATION_ENABLED:
            # In privacy mode, only log metadata
            logger.info(
                f"[Result] id={result.id} "
                f"has_translation={result.translation_ru is not None} "
                f"has_suggestion={result.suggestion is not None}"
            )
        else:
            print("\n" + "=" * 60)
            log_sensitive(f"[EN]: {result.transcript}")
            print(f"🗣️  [EN]: {result.transcript}")
            if result.translation_ru:
                log_sensitive(f"[RU]: {result.translation_ru}")
                print(f"🇷🇺  [RU]: {result.translation_ru}")
            if result.suggestion:
                print("-" * 60)
                verify_mark = "⚠️ (VERIFY)" if result.suggestion.needs_verification else "✅"
                log_sensitive(f"[AI EN]: {result.suggestion.answer_en}")
                print(f"💡 [AI EN] {verify_mark}: {result.suggestion.answer_en}")
                print(f"💡 [AI RU]: {result.suggestion.answer_ru}")
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
        
        # Audio backend will be stopped by the capture thread's finally block
        # Wait for threads to exit
        for t in self._threads:
            t.join(timeout=2.0)
            
        # Threads are dead — now it is safe to touch VAD state
        last_phrase = self.vad.flush()
        if last_phrase:
            try:
                self.phrase_queue.put(last_phrase, timeout=1.0)
            except queue.Full:
                pass
                
        self.vad.reset()  # Reset VAD state on stop
