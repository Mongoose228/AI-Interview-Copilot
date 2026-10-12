import asyncio
import queue
import threading
import time
from typing import Literal

from .audio.dsp import float32_to_pcm16, to_mono
from .audio.dual_capture import DualAudioCapture
from .audio.soundcard_wasapi import SoundCardWASAPIBackend
from .config import config
from .logging_config import log_sensitive, logger
from .models import PipelineResult, StageTiming, SuggestionStatus, Transcript
from .stt.base import SttError
from .stt.dual_path import DualPathSttEngine
from .stt.factory import build_stt_engine
from .stt.keyterms import build_initial_prompt, extract_keyterms
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
        self._mic_speaking_until = 0.0

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

        self.phrase_queue = queue.Queue(maxsize=20)
        self.partial_phrase_queue = queue.Queue(maxsize=1)
        self.transcript_queue = None
        self._loop = None
        self._stt_busy = threading.Event()
        self._phrase_state_lock = threading.Lock()
        self._final_phrase_ids: set = set()

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = paused
        self._threads = []
        self._device_id: str | None = None
        self._restart_capture = threading.Event()

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

    def _enqueue_partial(self, phrase) -> None:
        """Keep only the newest partial snapshot."""
        try:
            self.partial_phrase_queue.put_nowait(phrase)
        except queue.Full:
            try:
                self.partial_phrase_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.partial_phrase_queue.put_nowait(phrase)
            except queue.Full:
                pass

    def _enqueue_final(self, phrase) -> None:
        """Final phrases use their own queue and are never displaced by partials."""
        with self._phrase_state_lock:
            self._final_phrase_ids.add(phrase.id)
        try:
            queued_partial = self.partial_phrase_queue.get_nowait()
            if queued_partial.id != phrase.id:
                self._enqueue_partial(queued_partial)
        except queue.Empty:
            pass
        while not self._stop_event.is_set():
            try:
                self.phrase_queue.put(phrase, timeout=0.2)
                return
            except queue.Full:
                logger.warning("Final phrase queue full; waiting for STT worker.")

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

    def _on_streaming_transcript(self, transcript: Transcript) -> None:
        if transcript.speaker == "candidate":
            return
        if (
            (transcript.text_en or transcript.is_final)
            and self._loop
            and self.transcript_queue
        ):
            if transcript.is_final and transcript.speech_ended_at:
                logger.info(
                    "[Latency] speech_end_to_final_stt_ready="
                    f"{max(0.0, transcript.stt_ended_at - transcript.speech_ended_at):.3f}s"
                )
            self._loop.call_soon_threadsafe(
                self.transcript_queue.put_nowait, transcript
            )

    def clear_history(self):
        self.transcript_history.clear()

    # Audio objects are stopped only by the capture thread (WASAPI/COM isn't
    # safe to tear down cross-thread); these just signal it.
    def pause_capture(self):
        self._pause_event.set()
        if self._status_callback:
            self._status_callback("Paused")

    def resume_capture(self):
        self._pause_event.clear()
        self._restart_capture.set()
        if self._status_callback:
            self._status_callback("Listening…")

    def change_device(self, device_id: str | None):
        self._device_id = device_id
        self._restart_capture.set()

    def _process_mic_chunks(self) -> bool:
        """
        Drain candidate mic audio into a separate VAD (echo suppression only;
        candidate speech is never transcribed).
        Returns True while the candidate is speaking (or briefly after),
        so loopback STT/LLM can be suppressed against speaker echo.
        """
        if self.mic_vad is None or not isinstance(self.audio, DualAudioCapture):
            return time.time() < self._mic_speaking_until

        try:
            mic_chunks = self.audio.drain_mic_chunks()
        except Exception as e:
            logger.debug(f"drain_mic_chunks failed: {e}")
            return time.time() < self._mic_speaking_until

        for mic_chunk in mic_chunks:
            try:
                mic_phrases = self.mic_vad.process_chunk(mic_chunk)
            except Exception as e:
                logger.debug(f"Mic VAD error: {e}")
                continue
            if self.mic_vad.is_speaking or mic_phrases:
                # Hold suppress a bit after speech so loopback echo dies out.
                self._mic_speaking_until = time.time() + 0.45

        if self.mic_vad.is_speaking:
            self._mic_speaking_until = max(self._mic_speaking_until, time.time() + 0.15)

        return time.time() < self._mic_speaking_until

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

            # Tracked locally: reload_api_keys() may flip _streaming_stt mid-loop.
            stream_open = False
            try:
                current_device = self._device_id
                self.audio.start(current_device)
                self.vad.reset()
                if self.mic_vad is not None:
                    self.mic_vad.reset()
                self._mic_speaking_until = 0.0
                retries = 0
                self._restart_capture.clear()

                if self._streaming_stt:
                    try:
                        self.stt.start_stream(self._on_streaming_transcript)
                        stream_open = True
                    except Exception as e:
                        logger.error(f"Failed to start STT stream: {e}")
                        self._streaming_stt = False
                        self._use_vad_partials = config.PARTIAL_TRANSCRIPTION_ENABLED

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

                    mic_speaking = self._process_mic_chunks()

                    # Feed cloud streaming STT (PCM16 @ 16 kHz mono) — loopback only.
                    # While the mic is active nothing is sent; Deepgram KeepAlive holds the socket.
                    if stream_open and not mic_speaking:
                        try:
                            mono = to_mono(chunk.data, mode=config.DSP_MONO_MODE)
                            self.stt.feed_audio(float32_to_pcm16(mono), chunk.sample_rate)
                        except Exception as e:
                            logger.debug(f"STT feed_audio error: {e}")

                    # VAD still runs in streaming mode so the orchestrator knows
                    # when speech is ongoing, but streaming owns finals and partials.
                    phrases = self.vad.process_chunk(chunk)
                    if stream_open:
                        continue

                    for p in phrases:
                        if mic_speaking and config.AUDIO_SUPPRESS_LOOPBACK_WHILE_MIC:
                            logger.debug(
                                "Suppressing loopback phrase while candidate mic active"
                            )
                            continue
                        self._enqueue_final(p)

                    if (
                        self._use_vad_partials
                        and not mic_speaking
                        and not self._stt_busy.is_set()
                        and self.phrase_queue.empty()
                    ):
                        partial = self.vad.get_partial_snapshot()
                        if partial:
                            self._enqueue_partial(partial)

            except Exception as e:
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
                except Exception as stop_err:
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
                if stream_open:
                    try:
                        self.stt.close_stream()
                    except Exception as e:
                        logger.debug(f"STT close_stream: {e}")
                try:
                    self.audio.stop()
                except Exception as stop_err:
                    logger.warning(f"Error ignored: {stop_err}")

        logger.info("Audio capture thread stopped.")

    def _stt_worker(self):
        logger.info("STT thread started.")
        while not self._stop_event.is_set():
            # Finals always win over partials.
            try:
                phrase = self.phrase_queue.get_nowait()
            except queue.Empty:
                try:
                    phrase = self.partial_phrase_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
            if phrase is None:
                break
            if not hasattr(phrase, "audio_data"):
                continue
            try:
                self._transcribe_phrase(phrase)
            except SttError as e:
                logger.error(f"STT worker error: {e}")
                self._emit_error(f"Transcription error: {e}", transient=True)
            except Exception as e:
                logger.exception(f"STT worker error: {e}")
                self._emit_error(f"Transcription error: {e}", transient=True)
            finally:
                if phrase.is_final:
                    with self._phrase_state_lock:
                        self._final_phrase_ids.discard(phrase.id)
        logger.info("STT thread stopped.")

    def _transcribe_phrase(self, phrase) -> None:
        if not phrase.is_final:
            with self._phrase_state_lock:
                if phrase.id in self._final_phrase_ids:
                    return
        if self._status_callback:
            self._status_callback("Transcribing…")
        self._stt_busy.set()
        try:
            if phrase.is_final:
                transcript = self.stt.transcribe(phrase)
            elif isinstance(self.stt, DualPathSttEngine):
                # Partials are UI-only: use the fast provisional model, never the final engine.
                transcript = self.stt.maybe_provisional(phrase)
            else:
                transcript = self.stt.transcribe(phrase)
        finally:
            self._stt_busy.clear()

        if (
            transcript is not None
            and (transcript.text_en or transcript.is_final)
            and self._loop
            and self.transcript_queue
        ):
            if transcript.is_final and transcript.speech_ended_at:
                logger.info(
                    "[Latency] speech_end_to_final_stt_ready="
                    f"{max(0.0, transcript.stt_ended_at - transcript.speech_ended_at):.3f}s"
                )
            self._loop.call_soon_threadsafe(self.transcript_queue.put_nowait, transcript)
        elif self._status_callback and not self._pause_event.is_set():
            self._status_callback("Listening…")

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
                ended_at=transcript.stt_ended_at
                or transcript.stt_started_at + transcript.stt_duration_s,
                duration_s=transcript.stt_duration_s,
            )
        ]

        translation_coro = None
        suggestion_coro = None
        tokens_received = False
        first_token_logged = False

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
            nonlocal tokens_received, first_token_logged
            tokens_received = True
            self._tokens_seen.add(transcript.phrase_id)
            if not first_token_logged:
                first_token_logged = True
                if transcript.speech_ended_at:
                    latency = max(0.0, time.time() - transcript.speech_ended_at)
                    logger.info(
                        f"[Latency] speech_end_to_first_llm_token={latency:.3f}s"
                    )
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

        async def _none():
            return None

        try:
            translation_res, suggestion_res = await asyncio.gather(
                translation_coro or _none(),
                suggestion_coro or _none(),
                return_exceptions=True,
            )
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

        if isinstance(translation_res, BaseException):
            logger.warning(f"Translation failed: {translation_res}")
        else:
            translation_ru = translation_res
        if isinstance(suggestion_res, BaseException):
            logger.error(f"Processing error: {suggestion_res}")
            self._emit_error(f"Processing error: {suggestion_res}", transient=True)
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
                "OpenRouter API key not set. Add it in Settings or .env.",
            )
        elif not active_profile:
            self._suggestions_disabled_reason = "no_profile"
            self._emit_error(
                "No candidate profile found. Create a .md file in the context directory.",
            )

        await self.suggester.warm_up()

        current_llm_task = None
        pending_transcripts: list[Transcript] = []
        debounce_task: asyncio.Task | None = None
        active_card_id = None
        latest_partial_revisions: dict = {}
        first_partial_cards: set = set()
        phrase_card_ids: dict = {}

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
            nonlocal current_llm_task, pending_transcripts, active_card_id

            await asyncio.sleep(config.LLM_DEBOUNCE_DELAY_S)
            # A new phrase may already be in progress before its first partial STT result.
            # Keep the current question open until VAD finalizes that phrase (bounded).
            deadline = time.monotonic() + config.LLM_MAX_WAIT_FOR_SPEECH_END_S
            while getattr(self.vad, "is_speaking", False) and time.monotonic() < deadline:
                await asyncio.sleep(0.05)

            if not pending_transcripts:
                return

            segments = pending_transcripts
            pending_transcripts = []
            card_id = active_card_id or segments[0].phrase_id
            active_card_id = None
            first_partial_cards.discard(card_id)

            merged_text = " ".join(t.text_en for t in segments)
            merged_transcript = Transcript(
                phrase_id=card_id,
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
                is_final=True,
                revision=max(t.revision for t in segments),
                speech_started_at=min(
                    (t.speech_started_at for t in segments if t.speech_started_at),
                    default=0.0,
                ),
                speech_ended_at=max(
                    (t.speech_ended_at for t in segments if t.speech_ended_at),
                    default=0.0,
                ),
                audio_duration_s=sum(t.audio_duration_s for t in segments),
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

            # Final upsert keeps the same card used by partial/final STT.
            if self._transcript_callback:
                initial_result = PipelineResult(
                    id=card_id,
                    transcript=merged_transcript.text_en,
                    translation_ru=None,
                    suggestion=None,
                    profile=profile,
                    created_at=time.time(),
                    suggestion_status=SuggestionStatus.PENDING,
                    is_provisional=False,
                )
                self._transcript_callback(initial_result)

            current_llm_task = asyncio.create_task(
                self._process_transcript(merged_transcript, profile, history_snapshot)
            )
            self._background_tasks.add(current_llm_task)

            def on_task_done(t):
                self._background_tasks.discard(t)
                self._tokens_seen.discard(card_id)
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

            # Dual-channel: never send candidate mic speech to the LLM path.
            if getattr(transcript, "speaker", "interviewer") == "candidate":
                continue

            try:
                if transcript.is_final and (
                    not transcript.text_en or not transcript.text_en.strip()
                ):
                    provisional_card_id = phrase_card_ids.pop(
                        transcript.phrase_id, None
                    )
                    latest_partial_revisions.pop(transcript.phrase_id, None)
                    if (
                        not pending_transcripts
                        and provisional_card_id is not None
                    ):
                        if self._transcript_remove_callback:
                            self._transcript_remove_callback(provisional_card_id)
                        first_partial_cards.discard(provisional_card_id)
                        if active_card_id == provisional_card_id:
                            active_card_id = None
                    continue

                if not transcript.text_en or not transcript.text_en.strip():
                    continue

                if not transcript.is_final:
                    previous_revision = latest_partial_revisions.get(
                        transcript.phrase_id, -1
                    )
                    if transcript.revision <= previous_revision:
                        continue
                    latest_partial_revisions[transcript.phrase_id] = transcript.revision

                    if active_card_id is None:
                        active_card_id = transcript.phrase_id
                    phrase_card_ids[transcript.phrase_id] = active_card_id

                    prefix = " ".join(t.text_en for t in pending_transcripts)
                    display_text = " ".join(
                        part for part in (prefix, transcript.text_en) if part
                    )
                    with self._profile_lock:
                        profile = self._active_profile
                    if self._transcript_callback:
                        self._transcript_callback(
                            PipelineResult(
                                id=active_card_id,
                                transcript=display_text,
                                translation_ru=None,
                                suggestion=None,
                                profile=profile,
                                created_at=time.time(),
                                suggestion_status=SuggestionStatus.PENDING,
                                is_provisional=True,
                            )
                        )
                    if active_card_id not in first_partial_cards:
                        first_partial_cards.add(active_card_id)
                        if transcript.speech_started_at:
                            latency = max(
                                0.0, time.time() - transcript.speech_started_at
                            )
                            logger.info(
                                "[Latency] "
                                f"speech_start_to_first_partial_visible={latency:.3f}s"
                            )
                    continue

                latest_partial_revisions.pop(transcript.phrase_id, None)
                phrase_card_ids.pop(transcript.phrase_id, None)
                if is_filler(transcript.text_en):
                    logger.debug(
                        f"Dropped filler from LLM processing: '{transcript.text_en}'"
                    )
                    if not pending_transcripts and active_card_id is not None:
                        if self._transcript_remove_callback:
                            self._transcript_remove_callback(active_card_id)
                        first_partial_cards.discard(active_card_id)
                        active_card_id = None
                    continue

                if active_card_id is None:
                    active_card_id = transcript.phrase_id
                pending_transcripts.append(transcript)

                final_text = " ".join(t.text_en for t in pending_transcripts)
                with self._profile_lock:
                    profile = self._active_profile
                if self._transcript_callback:
                    visible_at = time.time()
                    self._transcript_callback(
                        PipelineResult(
                            id=active_card_id,
                            transcript=final_text,
                            translation_ru=None,
                            suggestion=None,
                            profile=profile,
                            created_at=visible_at,
                            suggestion_status=SuggestionStatus.PENDING,
                            is_provisional=False,
                        )
                    )
                    if transcript.speech_ended_at:
                        logger.info(
                            "[Latency] "
                            "speech_end_to_final_visible="
                            f"{max(0.0, visible_at - transcript.speech_ended_at):.3f}s "
                            "final_stt_to_visible="
                            f"{max(0.0, visible_at - transcript.stt_ended_at):.3f}s"
                        )

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
            logger.info(f"[Timings] {' -> '.join(timing_parts)} | total={total:.2f}s")

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

        # Load profile for keyterms before capture starts.
        with self._profile_lock:
            if self._active_profile is None:
                self._active_profile = self.profile_mgr.load_active_profile()
            profile = self._active_profile
        if profile:
            self._apply_profile_to_stt(profile.content)

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

        try:
            self.phrase_queue.put_nowait(None)
        except queue.Full:
            pass

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

        while not self.phrase_queue.empty():
            try:
                self.phrase_queue.get_nowait()
            except queue.Empty:
                break
        while not self.partial_phrase_queue.empty():
            try:
                self.partial_phrase_queue.get_nowait()
            except queue.Empty:
                break
        with self._phrase_state_lock:
            self._final_phrase_ids.clear()
        self._threads = []
