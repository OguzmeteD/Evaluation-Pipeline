from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import env_loader


class EnvLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        env_loader._LOADED_ENV_FILES.clear()

    def test_load_project_env_reads_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "LANGFUSE_PUBLIC_KEY=test-public",
                        "LANGFUSE_SECRET_KEY='test-secret'",
                        'LANGFUSE_HOST="https://cloud.langfuse.com"',
                        "export DATABASE_URL=postgresql://user:pass@localhost:5432/app",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded_path = env_loader.load_project_env(env_path)
                self.assertEqual(loaded_path, env_path.resolve())
                self.assertEqual(os.environ["LANGFUSE_PUBLIC_KEY"], "test-public")
                self.assertEqual(os.environ["LANGFUSE_SECRET_KEY"], "test-secret")
                self.assertEqual(os.environ["LANGFUSE_HOST"], "https://cloud.langfuse.com")
                self.assertEqual(
                    os.environ["DATABASE_URL"],
                    "postgresql://user:pass@localhost:5432/app",
                )

    def test_load_project_env_does_not_override_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("LANGFUSE_PUBLIC_KEY=file-public\n", encoding="utf-8")
            with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "shell-public"}, clear=True):
                env_loader.load_project_env(env_path)
                self.assertEqual(os.environ["LANGFUSE_PUBLIC_KEY"], "shell-public")

    def test_load_project_env_falls_back_to_src_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            src_env_path = project_root / "src" / ".env"
            src_env_path.parent.mkdir(parents=True, exist_ok=True)
            src_env_path.write_text("LANGFUSE_SECRET_KEY=src-secret\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                loaded_path = env_loader.load_project_env(project_root / "missing.env")
                self.assertIsNone(loaded_path)
                env_loader._LOADED_ENV_FILES.clear()
                with patch("src.core.env_loader._default_env_paths", return_value=[project_root / ".env", src_env_path]):
                    loaded_path = env_loader.load_project_env()
                self.assertEqual(loaded_path, src_env_path.resolve())
                self.assertEqual(os.environ["LANGFUSE_SECRET_KEY"], "src-secret")


if __name__ == "__main__":
    unittest.main()
