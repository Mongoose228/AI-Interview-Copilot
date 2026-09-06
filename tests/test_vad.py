import uuid
from unittest.mock import MagicMock

import numpy as np

from interview_copilot.models import AudioChunk
from interview_copilot.vad.silero import SileroVAD


def test_vad_detection(mocker):
    mocker.patch("interview_copilot.vad.silero.Path.exists", return_value=True)
    mocker.patch("onnxruntime.InferenceSession")
    vad = SileroVAD()
    vad._vad_lib_available = True

    # Mock infer to return high prob (start), then silence
    vad._infer = MagicMock()
    # 512*40 = 20480 samples. That is 40 chunks of 512.
    # Return > threshold on 1st, < threshold on rest
    vad._infer.side_effect = [0.9] + [0.0]*39

    chunk = AudioChunk(uuid.uuid4(), np.zeros((512*40, 2), dtype=np.float32), 16000, 2, 0.0)
    phrases = vad.process_chunk(chunk)

    assert len(phrases) == 1
    assert phrases[0].duration_s >= 0.25 # Should be greater than VAD_MIN_SPEECH_MS

def test_vad_preroll(mocker):
    mocker.patch("interview_copilot.vad.silero.Path.exists", return_value=True)
    mocker.patch("onnxruntime.InferenceSession")
    vad = SileroVAD()
    vad._vad_lib_available = True
    vad._infer = MagicMock()
    vad._infer.return_value = 0.0

    chunk = AudioChunk(uuid.uuid4(), np.zeros((512*10, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk)

    # Pre-roll buffer should be populated
    assert len(vad._preroll_buffer) > 0

    vad._infer.side_effect = [0.9, 0.0]
    chunk = AudioChunk(uuid.uuid4(), np.zeros((512*2, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk)

    # Preroll should be moved to phrase buffer
    assert len(vad._phrase_buffer) > 2

def test_vad_flush(mocker):
    mocker.patch("interview_copilot.vad.silero.Path.exists", return_value=True)
    mocker.patch("onnxruntime.InferenceSession")
    vad = SileroVAD()
    vad._vad_lib_available = True
    vad._infer = MagicMock()
    # 10 chunks: start on first, None for rest
    vad._infer.side_effect = [0.9] + [0.0]*9

    chunk = AudioChunk(uuid.uuid4(), np.zeros((512*10, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk)

    assert vad._is_speaking

    phrase = vad.flush()
    assert phrase is not None
    assert phrase.duration_s >= 0.25
    assert not vad._is_speaking

def test_preroll_cleared_after_speech_start(mocker):
    # Mock existence so it doesn't download
    mocker.patch("interview_copilot.vad.silero.Path.exists", return_value=True)
    mocker.patch("onnxruntime.InferenceSession")
    vad = SileroVAD()
    vad._vad_lib_available = True
    vad._infer = MagicMock()

    # 1. Provide silence to fill preroll
    vad._infer.return_value = 0.0
    chunk1 = AudioChunk(uuid.uuid4(), np.zeros((512*10, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk1)

    assert len(vad._preroll_buffer) > 0

    # 2. Start speech
    vad._infer.return_value = 0.9
    chunk2 = AudioChunk(uuid.uuid4(), np.zeros((512*1, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk2)

    # Preroll should have been moved and then cleared
    assert len(vad._phrase_buffer) > 1
    assert len(vad._preroll_buffer) == 0

def test_fast_subsequent_speech(mocker):
    mocker.patch("interview_copilot.vad.silero.Path.exists", return_value=True)
    mocker.patch("onnxruntime.InferenceSession")
    vad = SileroVAD()
    vad._vad_lib_available = True
    vad._infer = MagicMock()

    # Speech, then end speech
    vad._infer.side_effect = [0.9] + [0.0]*50 # Ends speech
    chunk = AudioChunk(uuid.uuid4(), np.zeros((512*20, 2), dtype=np.float32), 16000, 2, 0.0)
    phrases = vad.process_chunk(chunk)
    assert len(phrases) == 1

    # New speech immediately starts
    vad._infer.side_effect = [0.9] + [0.0]*50
    vad.process_chunk(chunk)
    # The new phrase buffer should not contain old preroll
    # Because preroll would only be populated if there was silence.
    # We just want to ensure it doesn't blow up.
    assert len(vad._preroll_buffer) == 0
