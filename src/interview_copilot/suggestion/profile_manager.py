import hashlib
import json
import os
import time

from ..config import config
from ..logging_config import logger
from ..models import ProfileSnapshot
from ..paths import get_profile_state_path


class ProfileManager:
    def __init__(self, context_dir: str | None = None, state_file: str | None = None):
        self._context_dir = context_dir or config.CONTEXT_DIR
        self._state_file = state_file or str(get_profile_state_path())

        if not os.path.exists(self._context_dir):
            os.makedirs(self._context_dir)

    def list_profiles(self) -> list[str]:
        """List all markdown files in the context directory."""
        if not os.path.exists(self._context_dir):
            return []

        profiles = []
        for filename in os.listdir(self._context_dir):
            if filename.endswith(".md"):
                profiles.append(filename[:-3])
        return sorted(profiles)

    def _get_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def load_profile(self, name: str) -> ProfileSnapshot | None:
        """Load a profile from disk and return a snapshot."""
        if not name or ".." in name or "/" in name or "\\" in name or os.sep in name:
            logger.error(f"Invalid profile name (path traversal attempt): {name}")
            return None

        file_path = os.path.join(self._context_dir, f"{name}.md")
        resolved = os.path.realpath(file_path)
        context_resolved = os.path.realpath(self._context_dir)
        if not resolved.startswith(context_resolved + os.sep):
            logger.error(f"Profile path escapes context directory: {resolved}")
            return None

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
                loaded_at=time.time(),
            )

            self._save_state(name)
            return snapshot
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error(f"Failed to load profile {name}: {e}")
            return None

    def _save_state(self, name: str):
        """Save the active profile name to state file."""
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump({"active_profile": name}, f)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error(f"Failed to save state: {e}")

    def load_active_profile(self) -> ProfileSnapshot | None:
        """Load the profile saved in the state file, or fallback to first available."""
        active_name = None
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    active_name = state.get("active_profile")
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(f"Error ignored: {e}")

        if active_name:
            snapshot = self.load_profile(active_name)
            if snapshot:
                return snapshot

        profiles = self.list_profiles()
        if profiles:
            real_profiles = [p for p in profiles if p != "example_profile"]
            if real_profiles:
                logger.warning(
                    f"Active profile not found or invalid. "
                    f"Falling back to {real_profiles[0]}"
                )
                return self.load_profile(real_profiles[0])
            logger.warning(
                "Only example_profile found. Skipping auto-load. "
                "Create your own profile in the context/ directory."
            )
            return None

        logger.warning("No profiles found. Suggestions will be disabled.")
        return None
