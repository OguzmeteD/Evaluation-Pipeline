from __future__ import annotations

import os
from pathlib import Path


_LOADED_ENV_FILES: set[Path] = set()


def load_project_env(env_path: str | Path | None = None) -> Path | None:
    """Load a project-local .env file into os.environ without overwriting existing keys."""
    candidate_paths = (
        [Path(env_path)]
        if env_path is not None
        else _default_env_paths()
    )
    for candidate in candidate_paths:
        path = candidate.expanduser().resolve()
        if path in _LOADED_ENV_FILES:
            return path if path.exists() else None
        if not path.exists():
            continue

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = _parse_env_value(value)

        _LOADED_ENV_FILES.add(path)
        return path

    return None


def _default_env_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    return [project_root / ".env", project_root / "src" / ".env"]


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in {"'", '"'} and len(value) >= 2:
        value = value[1:-1]
    return value
