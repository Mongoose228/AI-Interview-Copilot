"""Tests for ProfileManager: path traversal, auto-selection, loading."""
import os

import pytest

from interview_copilot.suggestion.profile_manager import ProfileManager


@pytest.fixture
def profile_dir(tmp_path):
    """Create a temporary context directory with test profiles."""
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "example_profile.md").write_text("# Example\nFake resume content", encoding="utf-8")
    (ctx / "real_candidate.md").write_text("# Real\nActual resume content", encoding="utf-8")
    return ctx


@pytest.fixture
def mgr(profile_dir, tmp_path, monkeypatch):
    """Create a ProfileManager pointing to the temp context dir."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr("interview_copilot.suggestion.profile_manager.config",
                        type("Config", (), {"CONTEXT_DIR": str(profile_dir)})())
    return ProfileManager()


class TestPathTraversal:
    def test_rejects_dotdot(self, mgr):
        assert mgr.load_profile("../etc/passwd") is None

    def test_rejects_forward_slash(self, mgr):
        assert mgr.load_profile("sub/dir") is None

    def test_rejects_backslash(self, mgr):
        assert mgr.load_profile("sub\\dir") is None

    def test_rejects_empty_name(self, mgr):
        assert mgr.load_profile("") is None

    def test_accepts_valid_name(self, mgr):
        result = mgr.load_profile("real_candidate")
        assert result is not None
        assert result.name == "real_candidate"
        assert "Actual resume content" in result.content


class TestAutoSelection:
    def test_skips_example_profile(self, mgr):
        """When no active profile saved, should NOT auto-select example_profile."""
        # Remove state file if it exists
        if os.path.exists(mgr._state_file):
            os.remove(mgr._state_file)

        result = mgr.load_active_profile()
        # Should fall back to real_candidate, NOT example_profile
        assert result is not None
        assert result.name == "real_candidate"

    def test_returns_none_when_only_example(self, profile_dir, tmp_path, monkeypatch):
        """When only example_profile exists, should return None."""
        # Remove the real profile
        (profile_dir / "real_candidate.md").unlink()

        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata2"))
        monkeypatch.setattr("interview_copilot.suggestion.profile_manager.config",
                            type("Config", (), {"CONTEXT_DIR": str(profile_dir)})())
        mgr = ProfileManager()

        result = mgr.load_active_profile()
        assert result is None

    def test_returns_none_when_no_profiles(self, tmp_path, monkeypatch):
        """When no profiles exist at all, should return None."""
        empty_ctx = tmp_path / "empty_context"
        empty_ctx.mkdir()

        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata3"))
        monkeypatch.setattr("interview_copilot.suggestion.profile_manager.config",
                            type("Config", (), {"CONTEXT_DIR": str(empty_ctx)})())
        mgr = ProfileManager()

        result = mgr.load_active_profile()
        assert result is None

    def test_loads_saved_active_profile(self, mgr):
        """When a valid profile is saved in state, it should be loaded."""
        # First, load a profile (which saves state)
        mgr.load_profile("real_candidate")

        # Now load_active_profile should return the saved one
        result = mgr.load_active_profile()
        assert result is not None
        assert result.name == "real_candidate"


class TestProfileListing:
    def test_list_profiles(self, mgr, profile_dir):
        profiles = mgr.list_profiles()
        assert "example_profile" in profiles
        assert "real_candidate" in profiles
        assert len(profiles) == 2

    def test_nonexistent_profile(self, mgr):
        result = mgr.load_profile("nonexistent_profile")
        assert result is None
