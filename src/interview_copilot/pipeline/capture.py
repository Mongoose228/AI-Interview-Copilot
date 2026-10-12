"""Capture thread: audio device → DSP'd chunks → VAD phrases / streaming STT."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..audio.dsp import float32_to_pcm16, to_mono
from ..audio.dual_capture import DualAudioCapture
from ..config import config
from ..logging_config import logger
from ..models import AudioChunk

if TYPE_CHECKING:
    from .core import InterviewPipeline

_MAX_AUDIO_RETRIES = 5
_AUDIO_RETRY_BASE_DELAY = 1.0
# Keep loopback suppressed briefly after the candidate stops so speaker echo dies out.
_MIC_HOLD_AFTER_SPEECH_S = 0.45
_MIC_HOLD_WHILE_SPEAKING_S = 0.15


class _GiveUp(Exception):
    """Audio retries exhausted; the capture thread should exit."""


class CaptureWorker:
    """
    Owns the audio device for the lifetime of the capture thread. Audio objects
    are started and stopped only here (WASAPI/COM isn't safe to tear down
    cross-thread); other threads request pause/restart via the pipeline's events.

    Reads ``audio``/``vad``/``stt`` from the pipeline on each use so components
    swapped at runtime (tests, key reload) are picked up.
    """

    def __init__(self, pipeline: InterviewPipeline):
        self._p = pipeline
        self._mic_speaking_until = 0.0

    def run(self, device_id: str | None = None) -> None:
        p = self._p
        logger.info("Audio capture thread started.")
        if p._vad_init_error:
            logger.error("VAD unavailable, capture thread cannot produce results.")
            p._emit_error(
                "VAD model failed to load. "
                "Speech detection is disabled. Check logs for details."
            )
            return

        p._device_id = device_id
        retries = 0
        while not p._stop_event.is_set():
            if p._pause_event.is_set():
                p._stop_event.wait(0.2)
                continue

            # Tracked locally: reload_api_keys() may flip _streaming_stt mid-loop.
            stream_open = False
            try:
                stream_open = self._open()
                retries = 0
                self._pump(stream_open)
            except Exception as e:
                if p._stop_event.is_set() or p._pause_event.is_set():
                    break
                if p._restart_capture.is_set():
                    # Device change raced with a read; reopen without counting a retry.
                    p._restart_capture.clear()
                    continue
                try:
                    retries = self._wait_before_retry(e, retries)
                except _GiveUp:
                    break
            finally:
                self._close(stream_open)

        logger.info("Audio capture thread stopped.")

    def _open(self) -> bool:
        """Start audio, reset VADs and open the STT stream; returns stream_open."""
        p = self._p
        p.audio.start(p._device_id)
        p.vad.reset()
        if p.mic_vad is not None:
            p.mic_vad.reset()
        self._mic_speaking_until = 0.0
        p._restart_capture.clear()

        stream_open = False
        if p._streaming_stt:
            try:
                p.stt.start_stream(p._on_streaming_transcript)
                stream_open = True
            except Exception as e:
                logger.error(f"Failed to start STT stream: {e}")
                p._streaming_stt = False
                p._use_vad_partials = config.PARTIAL_TRANSCRIPTION_ENABLED

        if p._audio_status_callback:
            p._audio_status_callback(True, "Audio capture active")
        p._emit_status("Listening…")
        return stream_open

    def _close(self, stream_open: bool) -> None:
        p = self._p
        if stream_open:
            try:
                p.stt.close_stream()
            except Exception as e:
                logger.debug(f"STT close_stream: {e}")
        try:
            p.audio.stop()
        except Exception as stop_err:
            logger.warning(f"Error ignored: {stop_err}")

    def _pump(self, stream_open: bool) -> None:
        """Read chunks until stop, pause or restart is requested."""
        p = self._p
        while not p._stop_event.is_set():
            if p._pause_event.is_set() or p._restart_capture.is_set():
                return
            try:
                chunk = p.audio.read_chunk()
            except RuntimeError:
                if p._stop_event.is_set() or p._pause_event.is_set():
                    return
                raise
            self._handle_chunk(chunk, stream_open)

    def _handle_chunk(self, chunk: AudioChunk, stream_open: bool) -> None:
        p = self._p
        mic_speaking = self._process_mic_chunks()

        # Feed cloud streaming STT (PCM16 @ 16 kHz mono) — loopback only.
        # While the mic is active nothing is sent; Deepgram KeepAlive holds the socket.
        if stream_open and not mic_speaking:
            try:
                mono = to_mono(chunk.data, mode=config.DSP_MONO_MODE)
                p.stt.feed_audio(float32_to_pcm16(mono), chunk.sample_rate)
            except Exception as e:
                logger.debug(f"STT feed_audio error: {e}")

        # VAD still runs in streaming mode so the orchestrator knows
        # when speech is ongoing, but streaming owns finals and partials.
        phrases = p.vad.process_chunk(chunk)
        if stream_open:
            return

        for phrase in phrases:
            if mic_speaking and config.AUDIO_SUPPRESS_LOOPBACK_WHILE_MIC:
                logger.debug("Suppressing loopback phrase while candidate mic active")
                continue
            p.queues.put_final(phrase)

        if p._use_vad_partials and not mic_speaking and p.queues.wants_partial():
            partial = p.vad.get_partial_snapshot()
            if partial:
                p.queues.put_partial(partial)

    def _process_mic_chunks(self) -> bool:
        """
        Drain candidate mic audio into a separate VAD (echo suppression only;
        candidate speech is never transcribed).
        Returns True while the candidate is speaking (or briefly after),
        so loopback STT/LLM can be suppressed against speaker echo.
        """
        p = self._p
        if p.mic_vad is None or not isinstance(p.audio, DualAudioCapture):
            return time.time() < self._mic_speaking_until

        try:
            mic_chunks = p.audio.drain_mic_chunks()
        except Exception as e:
            logger.debug(f"drain_mic_chunks failed: {e}")
            return time.time() < self._mic_speaking_until

        for mic_chunk in mic_chunks:
            try:
                mic_phrases = p.mic_vad.process_chunk(mic_chunk)
            except Exception as e:
                logger.debug(f"Mic VAD error: {e}")
                continue
            if p.mic_vad.is_speaking or mic_phrases:
                self._mic_speaking_until = time.time() + _MIC_HOLD_AFTER_SPEECH_S

        if p.mic_vad.is_speaking:
            self._mic_speaking_until = max(
                self._mic_speaking_until, time.time() + _MIC_HOLD_WHILE_SPEAKING_S
            )

        return time.time() < self._mic_speaking_until

    def _wait_before_retry(self, error: Exception, retries: int) -> int:
        """Back off after a capture failure; raises _GiveUp when retries run out."""
        p = self._p
        logger.error(f"Capture worker error: {error}")
        retries += 1
        if retries > _MAX_AUDIO_RETRIES:
            logger.error(
                f"Audio capture failed after {_MAX_AUDIO_RETRIES} retries. Giving up."
            )
            if p._audio_status_callback:
                p._audio_status_callback(False, f"Audio device lost: {error}")
            p._emit_error(f"Audio device lost after retries: {error}")
            raise _GiveUp

        delay = _AUDIO_RETRY_BASE_DELAY * (2 ** (retries - 1))
        logger.warning(
            f"Audio capture failed, retrying in {delay:.1f}s "
            f"(attempt {retries}/{_MAX_AUDIO_RETRIES})"
        )
        if p._audio_status_callback:
            p._audio_status_callback(
                False, f"Reconnecting... ({retries}/{_MAX_AUDIO_RETRIES})"
            )
        # Release the device during the back-off; run()'s finally only runs afterwards.
        try:
            p.audio.stop()
        except Exception as stop_err:
            logger.warning(f"Error ignored: {stop_err}")
        p._stop_event.wait(delay)
        return retries
