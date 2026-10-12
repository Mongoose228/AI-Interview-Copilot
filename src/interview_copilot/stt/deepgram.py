"""Deepgram STT: prerecorded HTTP + live WebSocket streaming."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from urllib.parse import urlencode

import httpx
import numpy as np

from ..audio.dsp import float32_to_pcm16
from ..config import config
from ..logging_config import logger
from ..models import SpeechPhrase, Transcript
from .base import SttEngine, SttError, TranscriptCallback

# WebSocketApp.send defaults to text frames; Deepgram needs audio as binary frames.
_OPCODE_BINARY = 0x2


class DeepgramEngine(SttEngine):
    # Deepgram closes the stream after ~10 s without audio or KeepAlive.
    _KEEPALIVE_INTERVAL_S = 5.0
    _KEEPALIVE_CHECK_S = 0.5
    _RECONNECT_BASE_DELAY_S = 1.0
    _RECONNECT_MAX_DELAY_S = 30.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ):
        self._api_key = api_key if api_key is not None else config.DEEPGRAM_API_KEY
        self._model = model or config.DEEPGRAM_MODEL
        self._language = language or config.DEEPGRAM_LANGUAGE
        self._keyterms: list[str] = []
        self._lock = threading.Lock()
        self._http: httpx.Client | None = None

        self._ws = None
        self._ws_app = None
        self._ws_thread: threading.Thread | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._on_transcript: TranscriptCallback | None = None
        self._stream_stop = threading.Event()
        self._stream_phrase_id: uuid.UUID | None = None
        self._stream_revision = 0
        self._stream_speech_started = 0.0
        self._send_lock = threading.Lock()
        self._last_send = 0.0

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def supports_streaming(self) -> bool:
        return True

    def set_keyterms(self, terms: list[str]) -> None:
        self._keyterms = list(terms or [])[:50]

    def reload_api_key(self) -> bool:
        new_key = config.DEEPGRAM_API_KEY
        if new_key == self._api_key:
            return False
        # The HTTP client carries no auth state; a running stream must be restarted.
        self._api_key = new_key
        return True

    def _get_http_client(self) -> httpx.Client:
        with self._lock:
            if self._http is None:
                self._http = httpx.Client(
                    timeout=httpx.Timeout(
                        config.NETWORK_READ_TIMEOUT,
                        connect=config.NETWORK_CONNECT_TIMEOUT,
                    )
                )
            return self._http

    def warm_up(self) -> None:
        if not self.is_available:
            return
        # Cheap auth check; ignore failures (network may be flaky at boot).
        try:
            self._get_http_client().get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {self._api_key}"},
                timeout=config.NETWORK_CONNECT_TIMEOUT,
            )
        except Exception as e:
            logger.debug(f"Deepgram warm_up skipped: {e}")

    def _query_params(self, *, streaming: bool) -> dict:
        params: dict = {
            "model": self._model,
            "language": self._language,
            "punctuate": "true",
            "smart_format": "true",
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
        }
        if streaming:
            params["interim_results"] = "true"
            params["endpointing"] = str(config.DEEPGRAM_ENDPOINTING_MS)
            params["utterance_end_ms"] = str(config.DEEPGRAM_UTTERANCE_END_MS)
            params["vad_events"] = "true"
        for term in self._keyterms:
            # Deepgram accepts repeated keyterm query params.
            params.setdefault("keyterm", [])
            if isinstance(params["keyterm"], list):
                params["keyterm"].append(term)
        return params

    def _build_url(self, *, streaming: bool) -> str:
        params = self._query_params(streaming=streaming)
        # urlencode list values as repeated keys
        pairs: list[tuple[str, str]] = []
        for k, v in params.items():
            if isinstance(v, list):
                for item in v:
                    pairs.append((k, str(item)))
            else:
                pairs.append((k, str(v)))
        qs = urlencode(pairs)
        host = "wss" if streaming else "https"
        return f"{host}://api.deepgram.com/v1/listen?{qs}"

    def transcribe(self, phrase: SpeechPhrase) -> Transcript:
        started = time.time()
        if not self.is_available:
            return self._empty(phrase, started, "Deepgram API key not set")

        audio = np.asarray(phrase.audio_data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        pcm = float32_to_pcm16(audio)
        url = self._build_url(streaming=False)
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/raw",
        }
        try:
            resp = self._get_http_client().post(url, content=pcm, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.error(f"Deepgram prerecorded failed: {e}")
            raise SttError(f"Deepgram prerecorded failed: {e}") from e
        text, conf = self._parse_prerecorded(payload)
        ended = time.time()
        return Transcript(
            phrase_id=phrase.id,
            text_en=text,
            language=self._language,
            confidence=conf,
            stt_duration_s=ended - started,
            speaker=getattr(phrase, "speaker", "interviewer"),
            stt_started_at=started,
            stt_ended_at=ended,
            is_final=phrase.is_final,
            revision=phrase.revision,
            speech_started_at=phrase.captured_at,
            speech_ended_at=phrase.vad_end_at,
            audio_duration_s=phrase.duration_s,
        )

    @staticmethod
    def _parse_prerecorded(payload: dict) -> tuple[str, float]:
        try:
            alt = payload["results"]["channels"][0]["alternatives"][0]
            return (alt.get("transcript") or "").strip(), float(alt.get("confidence") or 0.0)
        except (KeyError, IndexError, TypeError, ValueError):
            return "", 0.0

    def start_stream(self, on_transcript: TranscriptCallback) -> None:
        if not self.is_available:
            raise RuntimeError("Deepgram API key not set")
        self.close_stream()
        self._on_transcript = on_transcript
        self._stream_stop.clear()
        self._stream_phrase_id = uuid.uuid4()
        self._stream_revision = 0
        self._stream_speech_started = time.time()
        self._last_send = time.monotonic()
        self._ws_thread = threading.Thread(
            target=self._ws_loop, name="deepgram-ws", daemon=True
        )
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, name="deepgram-keepalive", daemon=True
        )
        self._ws_thread.start()
        self._keepalive_thread.start()

    def feed_audio(self, pcm16: bytes, sample_rate: int = 16000) -> None:
        if sample_rate != 16000:
            logger.warning("Deepgram feed_audio expects 16 kHz PCM16")
        ws = self._ws
        if not ws or self._stream_stop.is_set():
            return
        with self._send_lock:
            try:
                ws.send(pcm16, _OPCODE_BINARY)
                self._last_send = time.monotonic()
            except Exception as e:
                logger.debug(f"Deepgram send failed: {e}")

    def close_stream(self) -> None:
        self._stream_stop.set()
        ws = self._ws
        if ws is not None:
            with self._send_lock:
                try:
                    ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass
        ws_app = self._ws_app
        if ws_app is not None:
            try:
                ws_app.close()
            except Exception:
                pass
        for t in (self._ws_thread, self._keepalive_thread):
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=2.0)
        self._ws = None
        self._ws_app = None
        self._ws_thread = None
        self._keepalive_thread = None
        self._on_transcript = None

    def _keepalive_loop(self) -> None:
        while not self._stream_stop.wait(self._KEEPALIVE_CHECK_S):
            ws = self._ws
            if ws is None:
                continue
            if time.monotonic() - self._last_send < self._KEEPALIVE_INTERVAL_S:
                continue
            with self._send_lock:
                try:
                    ws.send(json.dumps({"type": "KeepAlive"}))
                    self._last_send = time.monotonic()
                except Exception as e:
                    logger.debug(f"Deepgram KeepAlive failed: {e}")

    def _ws_loop(self) -> None:
        try:
            import websocket
        except ImportError as e:
            logger.error(
                "websocket-client is required for Deepgram streaming: "
                "pip install websocket-client"
            )
            raise RuntimeError("websocket-client not installed") from e

        headers = [f"Authorization: Token {self._api_key}"]
        delay = self._RECONNECT_BASE_DELAY_S

        def on_message(_ws, message: str):
            self._handle_message(message)

        def on_error(_ws, error):
            logger.error(f"Deepgram WS error: {error}")

        def on_close(_ws, *_args):
            logger.info("Deepgram WS closed")

        def on_open(ws):
            nonlocal delay
            logger.info("Deepgram WS connected")
            delay = self._RECONNECT_BASE_DELAY_S
            self._last_send = time.monotonic()
            self._ws = ws

        while not self._stream_stop.is_set():
            ws_app = websocket.WebSocketApp(
                self._build_url(streaming=True),
                header=headers,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            self._ws_app = ws_app
            # close_stream() may have run before ws_app was published; don't outlive it.
            if self._stream_stop.is_set():
                break
            try:
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"Deepgram WS loop failed: {e}")
            finally:
                self._ws = None
            if self._stream_stop.is_set():
                break
            cb = self._on_transcript
            if cb and self._stream_revision > 0:
                # Close the interrupted utterance so its provisional card doesn't linger.
                self._emit(cb, text="", is_final=True, confidence=0.0)
            logger.warning(f"Deepgram WS disconnected; reconnecting in {delay:.1f}s")
            if self._stream_stop.wait(delay):
                break
            delay = min(delay * 2, self._RECONNECT_MAX_DELAY_S)

    def _handle_message(self, message: str) -> None:
        cb = self._on_transcript
        if not cb:
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        msg_type = payload.get("type")
        if msg_type == "UtteranceEnd":
            # Force finalize current phrase id for orchestrator.
            self._emit(
                cb,
                text="",
                is_final=True,
                confidence=0.0,
                force_new_after=True,
            )
            return

        if msg_type not in (None, "Results"):
            return
        channel = payload.get("channel") or {}
        alts = channel.get("alternatives") or []
        if not alts:
            return
        text = (alts[0].get("transcript") or "").strip()
        conf = float(alts[0].get("confidence") or 0.0)
        is_final = bool(payload.get("is_final") or payload.get("speech_final"))
        if not text and not is_final:
            return
        self._emit(cb, text=text, is_final=is_final, confidence=conf)

    def _emit(
        self,
        cb: Callable[[Transcript], None],
        *,
        text: str,
        is_final: bool,
        confidence: float,
        force_new_after: bool = False,
    ) -> None:
        if self._stream_phrase_id is None:
            self._stream_phrase_id = uuid.uuid4()
            self._stream_speech_started = time.time()
            self._stream_revision = 0
        self._stream_revision += 1
        now = time.time()
        transcript = Transcript(
            phrase_id=self._stream_phrase_id,
            text_en=text,
            language=self._language,
            confidence=confidence,
            stt_duration_s=max(0.0, now - self._stream_speech_started),
            stt_started_at=self._stream_speech_started,
            stt_ended_at=now,
            is_final=is_final,
            revision=self._stream_revision,
            speech_started_at=self._stream_speech_started,
            speech_ended_at=now if is_final else 0.0,
            audio_duration_s=max(0.0, now - self._stream_speech_started),
        )
        try:
            cb(transcript)
        except Exception as e:
            logger.error(f"Deepgram transcript callback failed: {e}")
        if is_final or force_new_after:
            self._stream_phrase_id = uuid.uuid4()
            self._stream_speech_started = time.time()
            self._stream_revision = 0

    def _empty(self, phrase: SpeechPhrase, started: float, _reason: str) -> Transcript:
        return Transcript(
            phrase_id=phrase.id,
            text_en="",
            language=self._language,
            confidence=0.0,
            stt_duration_s=time.time() - started,
            speaker=getattr(phrase, "speaker", "interviewer"),
            stt_started_at=started,
            stt_ended_at=time.time(),
            is_final=phrase.is_final,
            revision=phrase.revision,
            speech_started_at=phrase.captured_at,
            speech_ended_at=phrase.vad_end_at,
            audio_duration_s=phrase.duration_s,
        )
