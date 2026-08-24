"""Evaluation metrics and a deterministic CPU scorer.

The starter shipped hard-coded scores (``1.0`` for every chosen answer, ``0.0`` for
every rejected one), which makes ``pairwise_accuracy`` report 100% no matter what
the data says. This module replaces that with a real -- if deliberately small --
scorer so the number carries information.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .schemas import PreferenceExample

_TOKEN_RE = re.compile(r"[a-z0-9']+")

#: A response with no scoreable tokens gets this log-probability.
_EMPTY_SCORE = -math.inf


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Deterministic and dependency-free by design."""
    return _TOKEN_RE.findall(text.lower())


class UnigramScorer:
    """A smoothed unigram language model used as a stand-in for a policy model.

    Scoring a completion with ``sum(log p(token))`` is exactly what a real policy
    model gives a DPO/ORPO trainer, only with a far weaker model behind it. That is
    enough to exercise the whole pipeline on CPU and -- unlike constant mock scores
    -- it can be wrong, which is the point of measuring it.
    """

    def __init__(self, counts: Mapping[str, int], vocab: Iterable[str], alpha: float = 1.0) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive (Laplace smoothing), got {alpha}")
        self.alpha = alpha
        self.counts: dict[str, int] = dict(counts)
        self.vocab: frozenset[str] = frozenset(vocab) | frozenset(self.counts)
        # +1 slot for out-of-vocabulary tokens.
        self.vocab_size = len(self.vocab) + 1
        self.total = sum(self.counts.values())
        self._log_denominator = math.log(self.total + self.alpha * self.vocab_size)

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        vocab_texts: Iterable[str] | None = None,
        alpha: float = 1.0,
    ) -> UnigramScorer:
        """Fit on ``texts``; take the vocabulary from ``vocab_texts`` when given.

        Sharing one vocabulary across several scorers keeps their log-probabilities
        on the same scale, which is required for the DPO comparison to be fair.
        """
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        vocab: set[str] = set(counts)
        if vocab_texts is not None:
            for text in vocab_texts:
                vocab.update(tokenize(text))
        return cls(counts, vocab, alpha=alpha)

    def token_logprob(self, token: str) -> float:
        return math.log(self.counts.get(token, 0) + self.alpha) - self._log_denominator

    def sequence_logprob(self, text: str) -> float:
        """Sum of token log-probabilities: the sequence-level log-prob DPO expects."""
        tokens = tokenize(text)
        if not tokens:
            return _EMPTY_SCORE
        return sum(self.token_logprob(token) for token in tokens)

    def mean_logprob(self, text: str) -> float:
        """Length-normalized log-probability, used as the default ranking score.

        Without normalization the sum grows more negative with every extra token, so
        the scorer would simply prefer whichever answer is shorter.
        """
        tokens = tokenize(text)
        if not tokens:
            return _EMPTY_SCORE
        return self.sequence_logprob(text) / len(tokens)


def pairwise_accuracy(
    examples: Sequence[PreferenceExample],
    chosen_scores: Sequence[float],
    rejected_scores: Sequence[float],
    tie_credit: float = 0.5,
) -> float:
    """Return the fraction of examples where the chosen answer outscores the rejected.

    Args:
        examples: The evaluated examples.
        chosen_scores: Score per chosen completion, aligned with ``examples``.
        rejected_scores: Score per rejected completion, aligned with ``examples``.
        tie_credit: Credit awarded when the two scores are exactly equal. The
            default ``0.5`` reflects that a tie is a coin flip; pass ``0.0`` to
            treat ties as losses.

    Returns:
        A value in ``[0, 1]``. Returns ``0.0`` for an empty input.

    Raises:
        ValueError: If the three sequences differ in length, or ``tie_credit`` is
            outside ``[0, 1]``.
    """
    if not 0.0 <= tie_credit <= 1.0:
        raise ValueError(f"tie_credit must be within [0, 1], got {tie_credit}")
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError(
            "score/example length mismatch: "
            f"examples={len(examples)}, chosen_scores={len(chosen_scores)}, "
            f"rejected_scores={len(rejected_scores)}"
        )
    if not examples:
        return 0.0

    credit = 0.0
    for chosen, rejected in zip(chosen_scores, rejected_scores, strict=True):
        if chosen > rejected:
            credit += 1.0
        elif chosen == rejected:
            credit += tie_credit
    return credit / len(examples)


def count_ties(chosen_scores: Sequence[float], rejected_scores: Sequence[float]) -> int:
    """Number of examples the scorer could not separate at all."""
    return sum(
        1 for c, r in zip(chosen_scores, rejected_scores, strict=True) if c == r
    )


def longer_response_baseline(examples: Sequence[PreferenceExample]) -> float:
    """Accuracy of the trivial policy 'always prefer the longer answer'.

    Reported as a sanity floor: if the real scorer cannot beat this, it has learned
    length, not preference.
    """
    return pairwise_accuracy(
        examples,
        [float(len(e.chosen)) for e in examples],
        [float(len(e.rejected)) for e in examples],
    )


def evaluate_preferences(
    examples: Sequence[PreferenceExample],
    validation_ratio: float = 0.25,
    seed: int = 42,
    beta: float = 0.1,
    lambda_orpo: float = 0.1,
    tie_credit: float = 0.5,
    smoothing_alpha: float = 1.0,
) -> dict[str, float | int]:
    """Fit the deterministic scorer on the train split and score the validation split.

    Two unigram scorers are fitted over a shared vocabulary:

    * **policy** -- fitted on the *chosen* answers only, so it has seen the
      preference signal;
    * **reference** -- fitted on chosen *and* rejected answers, standing in for the
      un-aligned model DPO anchors against.

    Scoring only the held-out split is what keeps the number honest: the policy
    never saw those prompts, and :func:`~preference_lab.data.split_by_prompt`
    guarantees no prompt straddles the two sides.

    Returns:
        A metrics mapping ready for :func:`write_metrics`.

    Raises:
        ValueError: If ``examples`` is empty or a completion yields no tokens.
    """
    from .data import split_by_prompt
    from .losses import dpo_loss, orpo_loss

    if not examples:
        raise ValueError("no examples to evaluate")

    train, validation = split_by_prompt(list(examples), validation_ratio, seed=seed)
    all_train_text = [e.chosen for e in train] + [e.rejected for e in train]

    policy = UnigramScorer.fit(
        (e.chosen for e in train), vocab_texts=all_train_text, alpha=smoothing_alpha
    )
    reference = UnigramScorer.fit(
        all_train_text, vocab_texts=all_train_text, alpha=smoothing_alpha
    )

    def scores(scorer: UnigramScorer, rows: Sequence[PreferenceExample]) -> tuple[
        list[float], list[float], list[float], list[float]
    ]:
        """Return (mean chosen, mean rejected, seq chosen, seq rejected)."""
        return (
            [scorer.mean_logprob(e.chosen) for e in rows],
            [scorer.mean_logprob(e.rejected) for e in rows],
            [scorer.sequence_logprob(e.chosen) for e in rows],
            [scorer.sequence_logprob(e.rejected) for e in rows],
        )

    val_chosen, val_rejected, val_seq_chosen, val_seq_rejected = scores(policy, validation)
    _, _, ref_seq_chosen, ref_seq_rejected = scores(reference, validation)
    _, _, train_seq_chosen, train_seq_rejected = scores(policy, train)
    _, _, train_ref_chosen, train_ref_rejected = scores(reference, train)

    def implicit_rewards(policy_seq: Sequence[float], ref_seq: Sequence[float]) -> list[float]:
        """DPO's implicit reward: ``beta * (log pi(y|x) - log pi_ref(y|x))``.

        This is the quantity DPO actually optimizes, so it is the right thing to
        rank by. Both terms sum over the *same* tokens, so the length component
        largely cancels -- unlike a raw policy log-probability, which is really a
        length detector in disguise.
        """
        return [beta * (p - r) for p, r in zip(policy_seq, ref_seq, strict=True)]

    val_reward_chosen = implicit_rewards(val_seq_chosen, ref_seq_chosen)
    val_reward_rejected = implicit_rewards(val_seq_rejected, ref_seq_rejected)
    train_reward_chosen = implicit_rewards(train_seq_chosen, train_ref_chosen)
    train_reward_rejected = implicit_rewards(train_seq_rejected, train_ref_rejected)

    for label, values in (
        ("policy", val_seq_chosen + val_seq_rejected),
        ("reference", ref_seq_chosen + ref_seq_rejected),
    ):
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{label} scorer produced a non-finite score (empty completion?)")

    policy_chosen = np.asarray(val_seq_chosen, dtype=np.float64)
    policy_rejected = np.asarray(val_seq_rejected, dtype=np.float64)
    ref_chosen = np.asarray(ref_seq_chosen, dtype=np.float64)
    ref_rejected = np.asarray(ref_seq_rejected, dtype=np.float64)

    margins = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    # SFT term: per-token negative log-likelihood of the chosen answer.
    sft_nll = -np.asarray(val_chosen, dtype=np.float64)

    return {
        "n_examples": len(examples),
        "n_train": len(train),
        "n_validation": len(validation),
        "validation_ratio": validation_ratio,
        "seed": seed,
        "beta": beta,
        "lambda_orpo": lambda_orpo,
        # Primary metric: ranking by DPO's implicit reward on the held-out split.
        "pairwise_accuracy": pairwise_accuracy(
            validation, val_reward_chosen, val_reward_rejected, tie_credit=tie_credit
        ),
        "pairwise_accuracy_train": pairwise_accuracy(
            train, train_reward_chosen, train_reward_rejected, tie_credit=tie_credit
        ),
        # Diagnostics: ranking by raw policy log-probability, with and without
        # length normalization. Both are expected to be worse -- they measure
        # fluency and length, not preference.
        "pairwise_accuracy_mean_logprob": pairwise_accuracy(
            validation, val_chosen, val_rejected, tie_credit=tie_credit
        ),
        "pairwise_accuracy_seq_logprob": pairwise_accuracy(
            validation, val_seq_chosen, val_seq_rejected, tie_credit=tie_credit
        ),
        "longer_response_baseline": longer_response_baseline(validation),
        "ties": count_ties(val_reward_chosen, val_reward_rejected),
        "dpo_loss": dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta),
        "orpo_loss": orpo_loss(sft_nll, policy_chosen, policy_rejected, lambda_orpo),
        "mean_reward_margin": float(np.mean(margins)),
    }


def write_metrics(metrics: Mapping[str, float | int], output_dir: str | Path) -> Path:
    """Write ``metrics`` to ``<output_dir>/metrics.json`` and return that path.

    The value type is widened from ``float`` to ``float | int`` so counts such as
    ``n_validation`` serialize as ``12`` rather than ``12.0``.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(dict(metrics), indent=2, sort_keys=True), encoding="utf-8")
    return out
