from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a mapping.

    ``yaml.safe_load`` is untyped and happily returns ``None`` for an empty file or
    a scalar for a malformed one, so the result is checked here rather than blowing
    up later with a confusing ``TypeError`` deep inside the CLI.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the document's top level is not a mapping.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        loaded: Any = yaml.safe_load(f)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(
            f"{config_path}: expected a YAML mapping at the top level, "
            f"got {type(loaded).__name__}"
        )
    return loaded
