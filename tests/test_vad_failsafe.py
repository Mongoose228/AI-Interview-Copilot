"""Tests that VAD failure is properly reported."""
from unittest.mock import MagicMock

from interview_copilot.vad.silero import SileroVAD


def test_vad_unavailable_property(mocker):
    """When ONNX loading fails, is_available should be False."""
    mocker.patch("interview_copilot.vad.silero.Path.exists", return_value=False)
    mocker.patch(
        "interview_copilot.vad.silero.urllib.request.urlopen",
        side_effect=OSError("network error"),
    )
    vad = SileroVAD()
    assert not vad.is_available


def test_vad_available_when_loaded(mocker):
    """When ONNX loads successfully, is_available should be True."""
    mocker.patch("interview_copilot.vad.silero.Path.exists", return_value=True)
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [
        MagicMock(name="input"),
        MagicMock(name="h"),
        MagicMock(name="c"),
        MagicMock(name="sr"),
    ]
    # Set .name attribute on each mock input
    for inp, n in zip(mock_session.get_inputs.return_value, ["input", "h", "c", "sr"]):
        inp.name = n
    mocker.patch("onnxruntime.InferenceSession", return_value=mock_session)
    vad = SileroVAD()
    assert vad.is_available
