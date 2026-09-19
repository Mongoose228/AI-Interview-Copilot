import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from interview_copilot.models import AudioChunk
from interview_copilot.vad.silero import SileroVAD


@pytest.fixture
def vad(mocker, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "silero_vad.onnx").write_bytes(b"fake")
    mocker.patch("interview_copilot.vad.silero.get_models_dir", return_value=models)
    mocker.patch("onnxruntime.InferenceSession")
    instance = SileroVAD()
    instance._vad_lib_available = True
    return instance


def test_vad_detection(vad):
    vad._infer = MagicMock()
    vad._infer.side_effect = [0.9] + [0.0] * 39

    chunk = AudioChunk(uuid.uuid4(), np.zeros((512 * 40, 2), dtype=np.float32), 16000, 2, 0.0)
    phrases = vad.process_chunk(chunk)

    assert len(phrases) == 1
    assert phrases[0].duration_s >= 0.25


def test_vad_preroll(vad):
    vad._infer = MagicMock()
    vad._infer.return_value = 0.0

    chunk = AudioChunk(uuid.uuid4(), np.zeros((512 * 10, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk)

    assert len(vad._preroll_buffer) > 0

    vad._infer.side_effect = [0.9, 0.0]
    chunk = AudioChunk(uuid.uuid4(), np.zeros((512 * 2, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk)

    assert len(vad._phrase_buffer) > 2


def test_vad_flush(vad):
    vad._infer = MagicMock()
    vad._infer.side_effect = [0.9] + [0.0] * 9

    chunk = AudioChunk(uuid.uuid4(), np.zeros((512 * 10, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk)

    assert vad._is_speaking

    phrase = vad.flush()
    assert phrase is not None
    assert phrase.duration_s >= 0.25
    assert not vad._is_speaking


def test_preroll_cleared_after_speech_start(vad):
    vad._infer = MagicMock()

    vad._infer.return_value = 0.0
    chunk1 = AudioChunk(uuid.uuid4(), np.zeros((512 * 10, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk1)

    assert len(vad._preroll_buffer) > 0

    vad._infer.return_value = 0.9
    chunk2 = AudioChunk(uuid.uuid4(), np.zeros((512 * 1, 2), dtype=np.float32), 16000, 2, 0.0)
    vad.process_chunk(chunk2)

    assert len(vad._phrase_buffer) > 1
    assert len(vad._preroll_buffer) == 0


def test_fast_subsequent_speech(vad):
    vad._infer = MagicMock()

    vad._infer.side_effect = [0.9] + [0.0] * 50
    chunk = AudioChunk(uuid.uuid4(), np.zeros((512 * 20, 2), dtype=np.float32), 16000, 2, 0.0)
    phrases = vad.process_chunk(chunk)
    assert len(phrases) == 1

    vad._infer.side_effect = [0.9] + [0.0] * 50
    vad.process_chunk(chunk)
    assert len(vad._preroll_buffer) == 0
