import json
import math
from pathlib import Path

import pytest

from preference_lab.data import load_jsonl
from preference_lab.evaluate import (
    UnigramScorer,
    count_ties,
    evaluate_preferences,
    longer_response_baseline,
    pairwise_accuracy,
    tokenize,
    write_metrics,
)
from preference_lab.schemas import PreferenceExample

SAMPLE = "data/sample_preferences.jsonl"


def make(n: int) -> list[PreferenceExample]:
    return [
        PreferenceExample(prompt=f"p{i}", chosen=f"good answer {i}", rejected=f"bad answer {i}")
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# pairwise_accuracy
# --------------------------------------------------------------------------- #

def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_counts_losses() -> None:
    assert pairwise_accuracy(make(2), [1.0, 0.0], [2.0, 1.0]) == 0.0
    assert pairwise_accuracy(make(2), [3.0, 0.0], [2.0, 1.0]) == 0.5


def test_ties_get_half_credit_by_default() -> None:
    assert pairwise_accuracy(make(2), [1.0, 5.0], [1.0, 1.0]) == 0.75


@pytest.mark.parametrize(("tie_credit", "expected"), [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_tie_credit_policy_is_configurable(tie_credit: float, expected: float) -> None:
    assert pairwise_accuracy(make(1), [1.0], [1.0], tie_credit=tie_credit) == expected


def test_pairwise_accuracy_rejects_bad_tie_credit() -> None:
    with pytest.raises(ValueError, match="tie_credit"):
        pairwise_accuracy(make(1), [1.0], [1.0], tie_credit=1.5)


def test_pairwise_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        pairwise_accuracy(make(2), [1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="length mismatch"):
        pairwise_accuracy(make(2), [1.0, 1.0], [0.0])


def test_pairwise_accuracy_of_empty_input_is_zero() -> None:
    assert pairwise_accuracy([], [], []) == 0.0


def test_count_ties() -> None:
    assert count_ties([1.0, 2.0, 3.0], [1.0, 5.0, 3.0]) == 2


def test_longer_response_baseline_on_the_sample_corpus() -> None:
    """In this corpus the chosen answer is always the longer one -- a real bias."""
    assert longer_response_baseline(load_jsonl(SAMPLE)) == 1.0


# --------------------------------------------------------------------------- #
# UnigramScorer
# --------------------------------------------------------------------------- #

def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Self-Attention, really?!") == ["self", "attention", "really"]


def test_scorer_is_deterministic() -> None:
    corpus = ["alpha beta gamma", "alpha beta"]
    a = UnigramScorer.fit(corpus).sequence_logprob("alpha gamma")
    b = UnigramScorer.fit(corpus).sequence_logprob("alpha gamma")
    assert a == b


def test_scorer_prefers_frequent_tokens() -> None:
    scorer = UnigramScorer.fit(["alpha alpha alpha beta"])
    assert scorer.token_logprob("alpha") > scorer.token_logprob("beta")
    assert scorer.token_logprob("beta") > scorer.token_logprob("never-seen")


def test_scorer_handles_out_of_vocabulary_tokens_without_infinities() -> None:
    scorer = UnigramScorer.fit(["alpha beta"])
    assert math.isfinite(scorer.sequence_logprob("zzz qqq"))


def test_sequence_logprob_is_the_sum_and_mean_is_normalized() -> None:
    scorer = UnigramScorer.fit(["alpha beta gamma"])
    text = "alpha beta"
    assert scorer.sequence_logprob(text) == pytest.approx(
        scorer.token_logprob("alpha") + scorer.token_logprob("beta")
    )
    assert scorer.mean_logprob(text) == pytest.approx(scorer.sequence_logprob(text) / 2)


def test_sequence_logprob_grows_more_negative_with_length() -> None:
    """The length bias that makes raw log-probability a bad ranking score."""
    scorer = UnigramScorer.fit(["alpha beta gamma delta"])
    assert scorer.sequence_logprob("alpha beta gamma") < scorer.sequence_logprob("alpha")


def test_scorer_rejects_non_positive_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        UnigramScorer.fit(["alpha"], alpha=0.0)


# --------------------------------------------------------------------------- #
# End-to-end evaluation
# --------------------------------------------------------------------------- #

def test_evaluate_preferences_reports_the_expected_metrics() -> None:
    metrics = evaluate_preferences(load_jsonl(SAMPLE), validation_ratio=0.25, seed=42)
    expected_keys = {
        "n_examples",
        "n_train",
        "n_validation",
        "validation_ratio",
        "seed",
        "beta",
        "lambda_orpo",
        "pairwise_accuracy",
        "pairwise_accuracy_train",
        "pairwise_accuracy_mean_logprob",
        "pairwise_accuracy_seq_logprob",
        "longer_response_baseline",
        "ties",
        "dpo_loss",
        "orpo_loss",
        "mean_reward_margin",
    }
    assert set(metrics) == expected_keys
    assert metrics["n_train"] + metrics["n_validation"] == metrics["n_examples"] == 24
    assert all(math.isfinite(float(v)) for v in metrics.values())


def test_evaluate_preferences_accuracy_is_measured_not_hardcoded() -> None:
    """The starter returned a constant 1.0; a real scorer must be able to be wrong."""
    metrics = evaluate_preferences(load_jsonl(SAMPLE), validation_ratio=0.25, seed=42)
    accuracy = float(metrics["pairwise_accuracy"])
    assert 0.0 <= accuracy <= 1.0
    assert accuracy != 1.0
    # ...but it must still clearly beat a coin flip on held-out prompts.
    assert accuracy > 0.5


def test_evaluate_preferences_beats_the_raw_logprob_diagnostics() -> None:
    metrics = evaluate_preferences(load_jsonl(SAMPLE), validation_ratio=0.25, seed=42)
    assert metrics["pairwise_accuracy"] > metrics["pairwise_accuracy_seq_logprob"]
    assert metrics["pairwise_accuracy"] > metrics["pairwise_accuracy_mean_logprob"]


def test_dpo_loss_beats_the_uninformative_reference() -> None:
    """loss < log 2 means the policy separates the pair better than the reference."""
    metrics = evaluate_preferences(load_jsonl(SAMPLE), validation_ratio=0.25, seed=42)
    assert float(metrics["dpo_loss"]) < math.log(2.0)
    assert float(metrics["mean_reward_margin"]) > 0.0


def test_evaluate_preferences_is_reproducible_and_seed_sensitive() -> None:
    examples = load_jsonl(SAMPLE)
    a = evaluate_preferences(examples, validation_ratio=0.25, seed=42)
    b = evaluate_preferences(examples, validation_ratio=0.25, seed=42)
    assert a == b
    c = evaluate_preferences(examples, validation_ratio=0.25, seed=7)
    assert c["pairwise_accuracy"] != a["pairwise_accuracy"] or c["dpo_loss"] != a["dpo_loss"]


def test_evaluate_preferences_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no examples"):
        evaluate_preferences([])


# --------------------------------------------------------------------------- #
# write_metrics
# --------------------------------------------------------------------------- #

def test_write_metrics_roundtrip(tmp_path: Path) -> None:
    out = write_metrics({"pairwise_accuracy": 0.75, "n_validation": 6}, tmp_path / "nested")
    assert out.name == "metrics.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {"pairwise_accuracy": 0.75, "n_validation": 6}
    # Counts must not be stringified as floats.
    assert isinstance(payload["n_validation"], int)


def test_write_metrics_creates_missing_directories(tmp_path: Path) -> None:
    out = write_metrics({"a": 1.0}, tmp_path / "deep" / "deeper")
    assert out.is_file()
