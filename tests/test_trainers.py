import json
import math
from pathlib import Path

import numpy as np
import pytest

from preference_lab.config import load_config
from preference_lab.trainers import PreferenceTrainer, TrainingConfig


def config(tmp_path: Path, **overrides: object) -> TrainingConfig:
    base = {"method": "dpo", "epochs": 40, "output_dir": tmp_path}
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


def test_config_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        TrainingConfig(method="rlhf")


def test_config_rejects_bad_schedule() -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(method="dpo", epochs=-1)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(method="dpo", learning_rate=0.0)


def test_config_from_local_yaml_matches_the_committed_file() -> None:
    cfg = TrainingConfig.from_mapping(load_config("configs/local.yaml"))
    assert cfg.method == "dpo"
    assert cfg.beta == 0.1
    assert cfg.seed == 42
    assert cfg.train_data == Path("data/sample_preferences.jsonl")
    assert cfg.output_dir == Path("outputs")


def test_scores_are_always_valid_log_probabilities() -> None:
    weights = np.array([1.0, -1.0, 0.0])
    features = np.array([[-5.0, -6.0, 0.2], [50.0, -50.0, 9.9], [-1e3, 1e3, 0.0]])
    scores = PreferenceTrainer._score_features(weights, features)
    assert np.all(scores <= 0.0)
    assert np.all(np.isfinite(scores))


def test_dpo_training_starts_at_log_two_and_descends(tmp_path: Path) -> None:
    """The reference is frozen at the initial policy, so epoch 0 must cost log 2."""
    trainer = PreferenceTrainer(config(tmp_path))
    trainer.train()

    assert float(trainer.history[0]["train_loss"]) == pytest.approx(math.log(2.0))
    losses = [float(row["train_loss"]) for row in trainer.history]
    assert losses[-1] < losses[0]
    assert losses == sorted(losses, reverse=True), "gradient descent must not increase the loss"


def test_orpo_training_descends(tmp_path: Path) -> None:
    trainer = PreferenceTrainer(config(tmp_path, method="orpo"))
    trainer.train()
    losses = [float(row["train_loss"]) for row in trainer.history]
    assert losses[-1] < losses[0]


def test_mock_method_measures_without_updating_weights(tmp_path: Path) -> None:
    trainer = PreferenceTrainer(config(tmp_path, method="mock"))
    before = trainer.weights.copy()
    trainer.train()
    assert np.array_equal(trainer.weights, before)
    losses = {float(row["train_loss"]) for row in trainer.history}
    assert len(losses) == 1


def test_training_is_reproducible(tmp_path: Path) -> None:
    a = PreferenceTrainer(config(tmp_path / "a"))
    b = PreferenceTrainer(config(tmp_path / "b"))
    a.train()
    b.train()
    assert np.array_equal(a.weights, b.weights)
    assert a.history == b.history


def test_training_writes_checkpoint_and_curve(tmp_path: Path) -> None:
    trainer = PreferenceTrainer(config(tmp_path, epochs=5))
    trainer.train()

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["method"] == "dpo"
    assert checkpoint["features"] == list(PreferenceTrainer.FEATURES)
    assert len(checkpoint["weights"]) == len(PreferenceTrainer.FEATURES)

    curve = json.loads((tmp_path / "training_metrics.json").read_text(encoding="utf-8"))
    assert len(curve) == 6  # epochs 0..5 inclusive
    assert {"epoch", "train_loss", "val_loss", "val_pairwise_accuracy"} == set(curve[0])


def test_zero_epochs_only_records_the_starting_point(tmp_path: Path) -> None:
    trainer = PreferenceTrainer(config(tmp_path, epochs=0))
    trainer.train()
    assert len(trainer.history) == 1
