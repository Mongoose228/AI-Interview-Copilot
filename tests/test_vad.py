import uuid
from unittest.mock import MagicMock
import numpy as np

from interview_copilot.vad.silero import SileroVAD
from interview_copilot.models import AudioChunk


def test_vad_detection():
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

def test_vad_preroll():
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

def test_vad_flush():
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
