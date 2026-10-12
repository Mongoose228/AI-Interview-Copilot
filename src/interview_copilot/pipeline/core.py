"""InterviewPipeline: wires components, owns shared state and thread lifecycle.

Threads and their modules:
- capture thread  → ``capture.CaptureWorker``  (audio, VAD, streaming STT feed)
- STT thread      → ``stt_worker.SttWorker``   (batch/partial transcription)
- event loop      → ``orchestrator.TranscriptOrchestrator`` (cards, debounce, LLM)
"""

import asyncio
import threading
from typing import Literal

from ..audio.dual_capture import DualAudioCapture
from ..audio.soundcard_wasapi import SoundCardWASAPIBackend
from ..config import config
from ..logging_config import logger
from ..models import Transcript
from ..stt.factory import build_stt_engine
from ..stt.keyterms import build_initial_prompt, extract_keyterms
from ..suggestion.openrouter import OpenRouterSuggester
from ..suggestion.profile_manager import ProfileManager
from ..translation.deepl import DeepLTranslator
from ..translation.nllb import NLLBTranslator
from ..vad.silero import SileroVAD
from .capture import CaptureWorker
from .orchestrator import TranscriptOrchestrator
from .phrase_queues import PhraseQueues
from .stt_worker import SttWorker


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
        if audio is not None:
            self.audio = audio
        elif config.AUDIO_DUAL_CHANNEL:
            self.audio = DualAudioCapture()
        else:
            self.audio = SoundCardWASAPIBackend()
        self.vad = vad or SileroVAD(speaker="interviewer")
        self._vad_init_error = not self.vad.is_available
        if self._vad_init_error:
            logger.error("VAD model failed to load. Speech detection will not work.")

        # Second VAD for candidate mic when dual-channel capture is active.
        self.mic_vad = SileroVAD(speaker="candidate") if config.AUDIO_DUAL_CHANNEL else None

        self.stt = stt or build_stt_engine()
        self._refresh_streaming_mode()

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

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = paused
        self._restart_capture = threading.Event()
        self._threads = []
        self._device_id: str | None = None

        self.queues = PhraseQueues(self._stop_event)
        self.transcript_queue = None
        self._loop = None

        self._transcript_callback = None
        self._transcript_remove_callback = None
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
        self._background_tasks: set = set()

    def set_callbacks(
        self,
        transcript_callback,
        suggestion_callback,
        token_callback,
        error_callback,
        audio_status_callback,
        status_callback=None,
        transcript_remove_callback=None,
    ):
        self._transcript_callback = transcript_callback
        self._suggestion_callback = suggestion_callback
        self._token_callback = token_callback
        self._error_callback = error_callback
        self._audio_status_callback = audio_status_callback
        self._status_callback = status_callback
        self._transcript_remove_callback = transcript_remove_callback

    def _emit_error(self, msg: str, *, transient: bool = False):
        if self._error_callback:
            try:
                self._error_callback(msg, transient=transient)
            except TypeError:
                self._error_callback(msg)

    def _emit_status(self, msg: str) -> None:
        if self._status_callback:
            self._status_callback(msg)

    # --- profile / keys ------------------------------------------------------------

    def set_active_profile(self, name: str) -> bool:
        snapshot = self.profile_mgr.load_profile(name)
        if not snapshot:
            return False
        with self._profile_lock:
            self._active_profile = snapshot
            if self._suggestions_disabled_reason == "no_profile":
                if self.suggester.is_configured:
                    self._suggestions_disabled_reason = "none"
        self._apply_profile_to_stt(snapshot.content)
        return True

    def _apply_profile_to_stt(self, profile_text: str) -> None:
        if not config.STT_KEYTERMS_FROM_PROFILE:
            return
        terms = extract_keyterms(profile_text)
        prompt = build_initial_prompt(profile_text)
        try:
            self.stt.set_keyterms(terms)
            self.stt.set_initial_prompt(prompt)
            logger.info(f"[STT] applied {len(terms)} keyterms from profile")
        except Exception as e:
            logger.debug(f"Failed to apply STT keyterms: {e}")

    def _refresh_streaming_mode(self) -> None:
        self._streaming_stt = bool(
            getattr(self.stt, "supports_streaming", lambda: False)()
            and self.stt.is_available
            and config.STT_BACKEND in {"deepgram"}
        )
        # Cloud streaming owns finals and interim partials; VAD phrases are not transcribed.
        self._use_vad_partials = (
            config.PARTIAL_TRANSCRIPTION_ENABLED and not self._streaming_stt
        )

    def reload_api_keys(self) -> None:
        """Rebuild key-dependent clients from current ``config``; safe from any thread."""
        suggester = OpenRouterSuggester()
        with self._profile_lock:
            has_profile = self._active_profile is not None
            old_suggester = self.suggester
            self.suggester = suggester
            if not suggester.is_configured:
                self._suggestions_disabled_reason = "no_key"
            elif not has_profile:
                self._suggestions_disabled_reason = "no_profile"
            else:
                self._suggestions_disabled_reason = "none"

        if config.TRANSLATION_BACKEND == "deepl" and (
            not getattr(self.translator, "is_available", False)
            or getattr(self.translator, "_api_key", None) != config.DEEPL_API_KEY
        ):
            self.translator = DeepLTranslator()

        stt_key_changed = False
        try:
            stt_key_changed = self.stt.reload_api_key()
        except Exception as e:
            logger.warning(f"Failed to reload STT API key: {e}")
        if stt_key_changed:
            was_streaming = self._streaming_stt
            self._refresh_streaming_mode()
            if was_streaming or self._streaming_stt:
                # The capture loop reopens the stream with the new key on restart.
                self._restart_capture.set()

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._schedule_key_swap, old_suggester, suggester)

    def _schedule_key_swap(self, old_suggester, new_suggester) -> None:
        coros = []
        if old_suggester is not new_suggester and hasattr(old_suggester, "aclose"):

            async def close_later():
                # An in-flight answer may still be streaming on the old client.
                await asyncio.sleep(config.NETWORK_READ_TIMEOUT + 5)
                await old_suggester.aclose()

            coros.append(close_later())
        if new_suggester.is_configured:
            coros.append(new_suggester.warm_up())
        for coro in coros:
            task = asyncio.ensure_future(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    # --- transcripts → event loop ------------------------------------------------

    def _publish_transcript(self, transcript: Transcript | None) -> bool:
        """Hand a transcript to the orchestrator from any thread; False if dropped."""
        if (
            transcript is None
            or not (transcript.text_en or transcript.is_final)
            or not self._loop
            or not self.transcript_queue
        ):
            return False
        if transcript.is_final and transcript.speech_ended_at:
            logger.info(
                "[Latency] speech_end_to_final_stt_ready="
                f"{max(0.0, transcript.stt_ended_at - transcript.speech_ended_at):.3f}s"
            )
        self._loop.call_soon_threadsafe(self.transcript_queue.put_nowait, transcript)
        return True

    def _on_streaming_transcript(self, transcript: Transcript) -> None:
        if transcript.speaker == "candidate":
            return
        self._publish_transcript(transcript)

    def clear_history(self):
        self.transcript_history.clear()

    # --- capture control (signals only; the capture thread acts on them) -----------

    def pause_capture(self):
        self._pause_event.set()
        self._emit_status("Paused")

    def resume_capture(self):
        self._pause_event.clear()
        self._restart_capture.set()
        self._emit_status("Listening…")

    def change_device(self, device_id: str | None):
        self._device_id = device_id
        self._restart_capture.set()

    # --- lifecycle -------------------------------------------------------------------

    async def _async_orchestrator(self):
        await TranscriptOrchestrator(self).run()

    async def start(self, device_id: str | None = None):
        self._stop_event.clear()
        self._pause_event.clear()
        self._loop = asyncio.get_running_loop()
        self.transcript_queue = asyncio.Queue()

        # Load profile for keyterms before capture starts.
        with self._profile_lock:
            if self._active_profile is None:
                self._active_profile = self.profile_mgr.load_active_profile()
            profile = self._active_profile
        if profile:
            self._apply_profile_to_stt(profile.content)

        t_cap = threading.Thread(
            target=CaptureWorker(self).run,
            args=(device_id,),
            name="audio-capture",
            daemon=True,
        )
        t_stt = threading.Thread(target=SttWorker(self).run, name="stt", daemon=True)

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
        self.queues.put_sentinel()

        # The capture thread stops its own audio/stream; the calls below are a safety net.
        for t in self._threads:
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning(f"Thread {t.name} did not terminate in time.")

        try:
            if self._streaming_stt:
                self.stt.close_stream()
        except Exception as e:
            logger.debug(f"STT close_stream during stop: {e}")

        try:
            self.audio.stop()
        except Exception as e:
            logger.warning(f"Error stopping audio during shutdown: {e}")

        # Workers are gone, so an in-progress phrase can't be transcribed; discard it.
        try:
            self.vad.flush()
        except Exception as e:
            logger.warning(f"VAD flush during stop failed: {e}")
        self.vad.reset()

        self.queues.clear()
        self._threads = []
