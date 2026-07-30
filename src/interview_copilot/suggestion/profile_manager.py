import hashlib
import json
import os
import time

from ..config import config
from ..logging_config import logger
from ..models import ProfileSnapshot


class ProfileManager:
    def __init__(self):
        self._context_dir = config.CONTEXT_DIR

        # Store state in %APPDATA% (or fallback to CWD) to avoid CWD dependency
        appdata = os.environ.get("APPDATA", ".")
        state_dir = os.path.join(appdata, "interview_copilot")
        if not os.path.exists(state_dir):
            os.makedirs(state_dir, exist_ok=True)
        self._state_file = os.path.join(state_dir, ".copilot_state.json")

        # Ensure context dir exists
        if not os.path.exists(self._context_dir):
            os.makedirs(self._context_dir)

    def list_profiles(self) -> list[str]:
        """List all markdown files in the context directory."""
        if not os.path.exists(self._context_dir):
            return []

        profiles = []
        for filename in os.listdir(self._context_dir):
            if filename.endswith(".md"):
                profiles.append(filename[:-3])  # Strip .md
        return profiles

    def _get_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def load_profile(self, name: str) -> ProfileSnapshot | None:
        """Load a profile from disk and return a snapshot."""
        file_path = os.path.join(self._context_dir, f"{name}.md")
        if not os.path.exists(file_path):
            logger.error(f"Profile {name} not found at {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            snapshot = ProfileSnapshot(
                name=name,
                content=content,
                content_hash=self._get_hash(content),
                version=1,
                loaded_at=time.time(),
            )

            # Save to state file
            self._save_state(name)
            return snapshot
        except Exception as e:
            logger.error(f"Failed to load profile {name}: {e}")
            return None

    def _save_state(self, name: str):
        """Save the active profile name to state file."""
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump({"active_profile": name}, f)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_active_profile(self) -> ProfileSnapshot | None:
        """Load the profile saved in the state file, or fallback to first available."""
        active_name = None
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    active_name = state.get("active_profile")
            except Exception:
                pass

        if active_name:
            snapshot = self.load_profile(active_name)
            if snapshot:
                return snapshot

        # Fallback to first available
        profiles = self.list_profiles()
        if profiles:
            logger.warning(f"Active profile not found or invalid. Falling back to {profiles[0]}")
            return self.load_profile(profiles[0])

        logger.warning("No profiles found. Suggestions will be disabled.")
        return None
