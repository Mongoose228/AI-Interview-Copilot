import sys
import uuid

from PySide6.QtWidgets import QApplication

from interview_copilot.gui.main_window import CopilotMainWindow
from interview_copilot.models import PipelineResult, SuggestionResult

# Need a QApplication instance to run GUI tests
app = QApplication.instance() or QApplication(sys.argv)

def test_card_ordering():
    window = CopilotMainWindow()
    
    # Create two transcripts
    res1 = PipelineResult(uuid.uuid4(), "First transcript", None, None, None)
    res2 = PipelineResult(uuid.uuid4(), "Second transcript", None, None, None)
    
    window.add_transcript(res1)
    window.add_transcript(res2)
    
    assert len(window._result_widgets) == 2
    assert window._result_widgets[0].phrase_id == res1.id
    assert window._result_widgets[1].phrase_id == res2.id
    
    # Suggestion for res1 arrives AFTER res2 was added
    res1_updated = PipelineResult(res1.id, "First transcript", None, SuggestionResult("Answer 1"), None)
    window.update_suggestion(res1_updated)
    
    # The order of widgets should NOT change
    assert window._result_widgets[0].phrase_id == res1.id
    assert window._result_widgets[1].phrase_id == res2.id
    
    # Verify the suggestion text was appended
    assert "Answer 1" in window._result_widgets[0].current_suggestion_text
