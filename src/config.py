"""Config loading helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import PROJECT_ROOT


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache ``config.yaml`` or an explicit YAML path."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    return (PROJECT_ROOT / relative).resolve()
