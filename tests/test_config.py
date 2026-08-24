from pathlib import Path

import pytest

from preference_lab.config import load_config


def test_load_committed_local_config() -> None:
    cfg = load_config("configs/local.yaml")
    assert cfg["seed"] == 42
    assert cfg["paths"]["train_data"] == "data/sample_preferences.jsonl"
    assert cfg["training"]["method"] in {"dpo", "orpo", "mock"}
    assert cfg["evaluation"]["validation_ratio"] == 0.25


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("configs/nope.yaml")


def test_empty_config_is_an_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == {}


def test_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scalar.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(TypeError, match="expected a YAML mapping"):
        load_config(path)
