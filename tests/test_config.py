import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mumu.config import Settings, load_system_prompt
from mumu.errors import ConfigurationError


class ConfigTests(unittest.TestCase):
    def test_text_mode_does_not_require_voice_credentials(self):
        with patch.dict(os.environ, {"API_KEY": "dummy-secret", "PROJECT_ID": "project", "IBM_CLOUD_URL": "https://example.com"}, clear=True):
            settings = Settings.from_environment()
        self.assertEqual(settings.voice_id, "")
        self.assertNotIn("dummy-secret", repr(settings))

    def test_missing_settings_are_actionable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "API_KEY"):
                Settings.from_environment()

    def test_prompt_requires_real_file_and_rejects_blank(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompt.txt"
            with self.assertRaises(ConfigurationError):
                load_system_prompt(path)
            path.write_text("   ", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_system_prompt(path)
            path.write_text("안녕", encoding="utf-8-sig")
            self.assertEqual(load_system_prompt(path), "안녕")


if __name__ == "__main__":
    unittest.main()
