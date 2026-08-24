"""A CPU-only mock trainer for the preference objectives.

This is deliberately **not** an LLM fine-tune. It optimizes a three-parameter
linear reranker on top of the unigram scorers from :mod:`preference_lab.evaluate`,
using the very same :func:`~preference_lab.losses.dpo_loss` /
:func:`~preference_lab.losses.orpo_loss` implemented in this lab. That is enough to
show the objectives actually descend on real data, without a GPU or a model
download. Swapping in a TRL-backed trainer later means replacing
:meth:`PreferenceTrainer._score_features` and nothing else.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .data import load_jsonl, split_by_prompt
from .evaluate import UnigramScorer, pairwise_accuracy
from .losses import dpo_loss, orpo_loss
from .schemas import PreferenceExample

#: Objectives the trainer knows how to descend on. ``mock`` only measures.
SUPPORTED_METHODS = ("dpo", "orpo", "mock")

#: Step size used by the central-difference gradient estimate.
_FD_EPS = 1e-5


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    seed: int = 42
    epochs: int = 300
    learning_rate: float = 1.0
    validation_ratio: float = 0.25
    train_data: Path = Path("data/sample_preferences.jsonl")
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {SUPPORTED_METHODS}, got {self.method!r}"
            )
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> TrainingConfig:
        """Build a config from a parsed ``configs/*.yaml`` mapping."""
        training: Mapping[str, Any] = cfg.get("training") or {}
        paths: Mapping[str, Any] = cfg.get("paths") or {}
        evaluation: Mapping[str, Any] = cfg.get("evaluation") or {}
        base = cls(method=str(training.get("method", "dpo")))
        return replace(
            base,
            beta=float(training.get("beta", base.beta)),
            lambda_orpo=float(training.get("lambda_orpo", base.lambda_orpo)),
            max_length=int(training.get("max_length", base.max_length)),
            batch_size=int(training.get("batch_size", base.batch_size)),
            epochs=int(training.get("epochs", base.epochs)),
            learning_rate=float(training.get("learning_rate", base.learning_rate)),
            seed=int(cfg.get("seed", base.seed)),
            validation_ratio=float(evaluation.get("validation_ratio", base.validation_ratio)),
            train_data=Path(paths.get("train_data", base.train_data)),
            output_dir=Path(paths.get("output_dir", base.output_dir)),
        )


class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""

    #: Feature names of the linear reranker, in weight order.
    FEATURES = ("policy_mean_logprob", "reference_mean_logprob", "length_per_100_tokens")

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        # Start at the DPO implicit-reward direction: policy up, reference down,
        # length ignored. Training then adjusts that prior from data.
        self.weights: npt.NDArray[np.float64] = np.array([1.0, -1.0, 0.0], dtype=np.float64)
        self.history: list[dict[str, float | int]] = []

    # ----------------------------------------------------------------- setup #

    def _prepare(self) -> tuple[
        Sequence[PreferenceExample],
        Sequence[PreferenceExample],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Load data, split by prompt, and pre-compute the feature matrices."""
        examples = load_jsonl(self.config.train_data)
        train, validation = split_by_prompt(
            examples, self.config.validation_ratio, seed=self.config.seed
        )
        corpus = [e.chosen for e in train] + [e.rejected for e in train]
        policy = UnigramScorer.fit((e.chosen for e in train), vocab_texts=corpus)
        reference = UnigramScorer.fit(corpus, vocab_texts=corpus)

        def features(rows: Sequence[PreferenceExample], field: str) -> npt.NDArray[np.float64]:
            matrix = [
                [
                    policy.mean_logprob(getattr(row, field)),
                    reference.mean_logprob(getattr(row, field)),
                    len(getattr(row, field).split()) / 100.0,
                ]
                for row in rows
            ]
            return np.asarray(matrix, dtype=np.float64)

        return (
            train,
            validation,
            features(train, "chosen"),
            features(train, "rejected"),
            features(validation, "chosen"),
            features(validation, "rejected"),
        )

    # --------------------------------------------------------------- scoring #

    @staticmethod
    def _score_features(
        weights: npt.NDArray[np.float64], features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Map features to a pseudo log-probability in ``(-inf, 0)``.

        Both objectives require log-probabilities, so the raw linear score is
        squashed through ``-softplus(-s)``, which is monotone in ``s``, strictly
        negative, and preserves the ordering the reranker learns.
        """
        raw = features @ weights
        softplus = np.maximum(-raw, 0.0) + np.log1p(np.exp(-np.abs(raw)))
        result: npt.NDArray[np.float64] = -softplus
        return result

    def _objective(
        self,
        weights: npt.NDArray[np.float64],
        chosen_features: npt.NDArray[np.float64],
        rejected_features: npt.NDArray[np.float64],
        reference_chosen: npt.NDArray[np.float64],
        reference_rejected: npt.NDArray[np.float64],
    ) -> float:
        chosen = self._score_features(weights, chosen_features)
        rejected = self._score_features(weights, rejected_features)
        if self.config.method == "orpo":
            return orpo_loss(-chosen, chosen, rejected, self.config.lambda_orpo)
        return dpo_loss(
            chosen, rejected, reference_chosen, reference_rejected, self.config.beta
        )

    # -------------------------------------------------------------- training #

    def train(self) -> None:
        """Train the policy.

        Descends the configured objective by gradient descent, using a central
        finite-difference gradient: with only three parameters this is cheaper to
        get right than a hand-derived analytic gradient, and exact to ~1e-9.

        Side effects are explicit -- ``<output_dir>/checkpoint.json`` (weights) and
        ``<output_dir>/training_metrics.json`` (per-epoch curve) are the only files
        written.
        """
        train, validation, train_c, train_r, val_c, val_r = self._prepare()
        if not train or not validation:
            raise ValueError("need a non-empty train and validation split to train")

        # The reference scores are frozen at initialization: that is what makes the
        # DPO anchor meaningful.
        ref_train_c = self._score_features(self.weights, train_c)
        ref_train_r = self._score_features(self.weights, train_r)
        ref_val_c = self._score_features(self.weights, val_c)
        ref_val_r = self._score_features(self.weights, val_r)

        self.history = []
        for epoch in range(self.config.epochs + 1):
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": self._objective(
                        self.weights, train_c, train_r, ref_train_c, ref_train_r
                    ),
                    "val_loss": self._objective(
                        self.weights, val_c, val_r, ref_val_c, ref_val_r
                    ),
                    "val_pairwise_accuracy": pairwise_accuracy(
                        validation,
                        list(self._score_features(self.weights, val_c)),
                        list(self._score_features(self.weights, val_r)),
                    ),
                }
            )
            if epoch == self.config.epochs or self.config.method == "mock":
                continue
            gradient = self._gradient(train_c, train_r, ref_train_c, ref_train_r)
            self.weights = self.weights - self.config.learning_rate * gradient

        self._save()

    def _gradient(
        self,
        chosen_features: npt.NDArray[np.float64],
        rejected_features: npt.NDArray[np.float64],
        reference_chosen: npt.NDArray[np.float64],
        reference_rejected: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        gradient = np.zeros_like(self.weights)
        for index in range(self.weights.size):
            step = np.zeros_like(self.weights)
            step[index] = _FD_EPS
            forward = self._objective(
                self.weights + step, chosen_features, rejected_features,
                reference_chosen, reference_rejected,
            )
            backward = self._objective(
                self.weights - step, chosen_features, rejected_features,
                reference_chosen, reference_rejected,
            )
            gradient[index] = (forward - backward) / (2.0 * _FD_EPS)
        return gradient

    # ---------------------------------------------------------------- output #

    def _save(self) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "method": self.config.method,
            "beta": self.config.beta,
            "lambda_orpo": self.config.lambda_orpo,
            "seed": self.config.seed,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "features": list(self.FEATURES),
            "weights": [float(w) for w in self.weights],
        }
        (output_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "training_metrics.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
