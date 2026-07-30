import asyncio
import queue
import threading
import time

from .audio.soundcard_wasapi import SoundCardWASAPIBackend
from .config import config
from .logging_config import log_sensitive, logger
from .models import PipelineResult, Transcript
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
        self.transcript_queue = queue.Queue(maxsize=20)

        self._stop_event = threading.Event()
        self._threads = []

        # Callbacks
        self._result_callback = None
        self._audio_status_callback = None
        self._error_callback = None

        # History (bounded)
        self.transcript_history: list[Transcript] = []

        # Track active suggestion task for cancellation
        self._current_suggestion_task: asyncio.Task | None = None

    def set_result_callback(self, callback):
        self._result_callback = callback

    def set_audio_status_callback(self, callback):
        """Callback for audio device status changes: callback(is_connected: bool, message: str)"""
        self._audio_status_callback = callback

    def set_error_callback(self, callback):
        self._error_callback = callback

    def _capture_and_vad_worker(self, device_id: str = None):
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

            except Exception as e:
                if self._stop_event.is_set():
                    break
                logger.error(f"Capture worker error: {e}")
                try:
                    self.audio.stop()
                except Exception:
                    pass

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
                except Exception:
                    pass

        logger.info("Audio capture thread stopped.")

    def _stt_worker(self):
        logger.info("STT thread started.")
        while not self._stop_event.is_set():
            try:
                phrase = self.phrase_queue.get(timeout=0.5)
                transcript = self.stt.transcribe(phrase)
                if transcript.text_en:
                    try:
                        self.transcript_queue.put(transcript, timeout=1.0)
                    except queue.Full:
                        logger.warning("Transcript queue is full, dropping transcript.")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"STT worker error: {e}")
                if self._error_callback:
                    self._error_callback(f"Transcription error: {e}")
        logger.info("STT thread stopped.")

    async def _process_transcript(self, transcript: Transcript, active_profile):
        """Process a single transcript: translate + get suggestion concurrently."""
        # Build tasks for concurrent execution
        translation_coro = None
        suggestion_coro = None

        if self.translator:
            translation_coro = asyncio.to_thread(
                self.translator.translate, transcript.text_en
            )

        if active_profile:
            suggestion_coro = self.suggester.get_suggestion(
                self.transcript_history, active_profile
            )

        # Run translation and suggestion concurrently
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
        except Exception as e:
            logger.error(f"Processing error: {e}")
            if self._error_callback:
                self._error_callback(f"Processing error: {e}")

        # Create final result
        result = PipelineResult(
            id=transcript.phrase_id,
            transcript=transcript.text_en,
            translation_ru=translation_ru,
            suggestion=suggestion,
            profile=active_profile,
            created_at=time.time(),
        )

        self._display_result(result)
        if self._result_callback:
            self._result_callback(result)

    async def _async_orchestrator(self):
        logger.info("Async orchestrator started.")
        active_profile = self.profile_mgr.load_active_profile()

        while not self._stop_event.is_set():
            try:
                # Use get_nowait to avoid blocking the event loop
                transcript = self.transcript_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            self.transcript_history.append(transcript)

            # Trim history to prevent unbounded growth
            if len(self.transcript_history) > _MAX_HISTORY:
                self.transcript_history = self.transcript_history[-_MAX_HISTORY:]

            # Cancel stale suggestion task if still running
            if self._current_suggestion_task and not self._current_suggestion_task.done():
                self._current_suggestion_task.cancel()
                try:
                    await self._current_suggestion_task
                except (asyncio.CancelledError, Exception):
                    pass

            # Launch processing as a task (non-blocking)
            self._current_suggestion_task = asyncio.create_task(
                self._process_transcript(transcript, active_profile)
            )

        # Cleanup: cancel any remaining task
        if self._current_suggestion_task and not self._current_suggestion_task.done():
            self._current_suggestion_task.cancel()

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

    async def start(self, device_id: str = None):
        self._stop_event.clear()

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
        
        # Stop audio backend BEFORE joining threads to unblock read_chunk
        try:
            self.audio.stop()
        except Exception:
            pass
            
        # Flush last phrase if we were speaking
        last_phrase = self.vad.flush()
        if last_phrase:
            try:
                self.phrase_queue.put(last_phrase, timeout=1.0)
            except queue.Full:
                pass
                
        self.vad.reset()  # Reset VAD state on stop
        
        for t in self._threads:
            # Short timeout, they should unblock now that audio is stopped
            t.join(timeout=2.0)
