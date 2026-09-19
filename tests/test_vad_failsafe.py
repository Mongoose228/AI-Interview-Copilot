"""Tests that VAD failure is properly reported."""
from unittest.mock import MagicMock

from interview_copilot.vad.silero import SileroVAD


def test_vad_unavailable_property(mocker, tmp_path):
    """When ONNX loading fails, is_available should be False."""
    models = tmp_path / "models"
    models.mkdir()
    mocker.patch("interview_copilot.vad.silero.get_models_dir", return_value=models)
    mocker.patch(
        "interview_copilot.vad.silero.urllib.request.urlopen",
        side_effect=OSError("network error"),
    )
    vad = SileroVAD()
    assert not vad.is_available


def test_vad_available_when_loaded(mocker, tmp_path):
    """When ONNX loads successfully, is_available should be True."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "silero_vad.onnx").write_bytes(b"fake")
    mocker.patch("interview_copilot.vad.silero.get_models_dir", return_value=models)
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [
        MagicMock(name="input"),
        MagicMock(name="h"),
        MagicMock(name="c"),
        MagicMock(name="sr"),
    ]
    for inp, n in zip(
        mock_session.get_inputs.return_value, ["input", "h", "c", "sr"], strict=True
    ):
        inp.name = n
    mocker.patch("onnxruntime.InferenceSession", return_value=mock_session)
    vad = SileroVAD()
    assert vad.is_available
