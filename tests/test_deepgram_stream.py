"""Deepgram streaming message parsing (no network)."""

import json
import uuid

from interview_copilot.stt.deepgram import DeepgramEngine


def test_deepgram_handle_results_interim_and_final():
    eng = DeepgramEngine(api_key="test-key")
    seen = []
    eng._on_transcript = seen.append
    eng._stream_phrase_id = uuid.uuid4()
    eng._stream_revision = 0
    eng._stream_speech_started = 1.0

    interim = {
        "type": "Results",
        "is_final": False,
        "channel": {"alternatives": [{"transcript": "hello", "confidence": 0.9}]},
    }
    eng._handle_message(json.dumps(interim))
    assert len(seen) == 1
    assert seen[0].text_en == "hello"
    assert seen[0].is_final is False

    final = {
        "type": "Results",
        "is_final": True,
        "speech_final": True,
        "channel": {
            "alternatives": [{"transcript": "hello world", "confidence": 0.95}]
        },
    }
    eng._handle_message(json.dumps(final))
    assert len(seen) == 2
    assert seen[1].is_final is True
    assert "world" in seen[1].text_en


def test_soniox_unavailable_without_key():
    from interview_copilot.stt.soniox import SonioxEngine

    eng = SonioxEngine(api_key=None)
    assert eng.is_available is False
