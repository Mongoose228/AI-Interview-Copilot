"""Tests for phrase filler detection."""
from interview_copilot.suggestion.phrase_filter import is_filler


class TestFillerDetection:
    def test_known_fillers(self):
        assert is_filler("uh") is True
        assert is_filler("okay") is True
        assert is_filler("um") is True
        assert is_filler("yeah") is True
        assert is_filler("i see") is True
        assert is_filler("got it") is True

    def test_case_insensitive(self):
        assert is_filler("Okay") is True
        assert is_filler("YEAH") is True
        assert is_filler("Uh") is True

    def test_with_punctuation(self):
        assert is_filler("okay.") is True
        assert is_filler("yeah!") is True
        assert is_filler("um...") is True

    def test_real_questions_not_filler(self):
        assert is_filler("What is polymorphism?") is False
        assert is_filler("Tell me about your experience with Python") is False
        assert is_filler("How do you handle deadlocks?") is False

    def test_short_but_not_filler(self):
        assert is_filler("hello world test") is False
        assert is_filler("binary tree") is False

    def test_empty_string(self):
        assert is_filler("") is True

    def test_whitespace_only(self):
        assert is_filler("   ") is True

    def test_longer_filler_phrases(self):
        assert is_filler("you know") is True
        assert is_filler("makes sense") is True
        assert is_filler("mm hmm") is True

    def test_four_words_not_filler(self):
        """Phrases longer than 3 words should never be considered filler."""
        assert is_filler("I see your point") is False
        assert is_filler("okay so what now") is False
