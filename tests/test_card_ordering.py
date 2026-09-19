import sys
import uuid

from PySide6.QtWidgets import QApplication

from interview_copilot.gui.main_window import CopilotMainWindow
from interview_copilot.models import PipelineResult, SuggestionResult, SuggestionStatus

app = QApplication.instance() or QApplication(sys.argv)


def test_card_ordering():
    window = CopilotMainWindow()

    res1 = PipelineResult(uuid.uuid4(), "First transcript", None, None, None)
    res2 = PipelineResult(uuid.uuid4(), "Second transcript", None, None, None)

    window.add_transcript(res1)
    window.add_transcript(res2)

    assert len(window._result_widgets) == 2
    assert window._result_widgets[0].phrase_id == res1.id
    assert window._result_widgets[1].phrase_id == res2.id

    res1_updated = PipelineResult(
        res1.id,
        "First transcript",
        None,
        SuggestionResult(answer_en="Answer 1"),
        None,
        suggestion_status=SuggestionStatus.OK,
    )
    window.update_suggestion(res1_updated)

    assert window._result_widgets[0].phrase_id == res1.id
    assert window._result_widgets[1].phrase_id == res2.id
    assert "Answer 1" in window._result_widgets[0].current_suggestion_text


def test_html_escape_in_transcript():
    window = CopilotMainWindow()
    res = PipelineResult(
        uuid.uuid4(),
        "Use <script>alert(1)</script> & co",
        None,
        None,
        None,
    )
    window.add_transcript(res)
    label_text = window._result_widgets[0].findChildren(
        __import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel
    )[0].text()
    assert "<script>" not in label_text
    assert "&lt;script&gt;" in label_text


def test_skipped_no_key_status():
    window = CopilotMainWindow()
    res = PipelineResult(uuid.uuid4(), "Question?", None, None, None)
    window.add_transcript(res)
    window.update_suggestion(
        PipelineResult(
            res.id,
            "Question?",
            None,
            None,
            None,
            suggestion_status=SuggestionStatus.SKIPPED_NO_KEY,
        )
    )
    assert "OPENROUTER_API_KEY" in window._result_widgets[0].lbl_ai.text()
