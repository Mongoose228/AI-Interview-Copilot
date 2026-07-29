import asyncio
import queue
import threading
import time

from .audio.soundcard_wasapi import SoundCardWASAPIBackend
from .config import config
from .logging_config import logger
from .models import PipelineResult, Transcript
from .stt.whisper_engine import WhisperEngine
from .suggestion.openrouter import OpenRouterSuggester
from .suggestion.profile_manager import ProfileManager
from .translation.deepl import DeepLTranslator
from .translation.nllb import NLLBTranslator
from .vad.silero import SileroVAD


class InterviewPipeline:
    def __init__(self):
        self.audio = SoundCardWASAPIBackend()
        self.vad = SileroVAD()
        self.stt = WhisperEngine()

        # Translation strategy
        self.translator = None
        if config.LOCAL_TRANSLATION_ENABLED:
            if config.NLLB_ENABLED:
                self.translator = NLLBTranslator()
            else:
                self.translator = DeepLTranslator()
                # Fallback to NLLB if DeepL fails (e.g. no key)
                if not getattr(self.translator, "_translator", None):
                    logger.info("DeepL not available, falling back to NLLB...")
                    self.translator = NLLBTranslator()

        self.suggester = OpenRouterSuggester()
        self.profile_mgr = ProfileManager()

        self.phrase_queue = queue.Queue(maxsize=20)
        self.transcript_queue = queue.Queue(maxsize=20)

        self._stop_event = threading.Event()
        self._threads = []

        # Callbacks
        self._result_callback = None

        # History
        self.transcript_history: list[Transcript] = []

    def set_result_callback(self, callback):
        self._result_callback = callback

    def _capture_and_vad_worker(self, device_id: str = None):
        logger.info("Audio capture thread started.")
        self.audio.start(device_id)
        try:
            while not self._stop_event.is_set():
                chunk = self.audio.read_chunk()
                phrases = self.vad.process_chunk(chunk)
                for p in phrases:
                    try:
                        self.phrase_queue.put(p, timeout=1.0)
                    except queue.Full:
                        logger.warning("Phrase queue is full, dropping phrase.")
        except Exception as e:
            logger.error(f"Capture worker error: {e}")
        finally:
            self.audio.stop()
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
        logger.info("STT thread stopped.")

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

            # 1. Translate
            translation_ru = None
            if self.translator:
                translation_ru = await asyncio.to_thread(
                    self.translator.translate, transcript.text_en
                )

            # 2. Suggestion
            suggestion = None
            if active_profile:
                suggestion = await self.suggester.get_suggestion(
                    self.transcript_history, active_profile
                )

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

        logger.info("Async orchestrator stopped.")

    def _display_result(self, result: PipelineResult):
        print("\n" + "=" * 60)
        print(f"🗣️  [EN]: {result.transcript}")
        if result.translation_ru:
            print(f"🇷🇺  [RU]: {result.translation_ru}")
        if result.suggestion:
            print("-" * 60)
            verify_mark = "⚠️ (VERIFY)" if result.suggestion.needs_verification else "✅"
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
        for t in self._threads:
            t.join(timeout=2.0)
