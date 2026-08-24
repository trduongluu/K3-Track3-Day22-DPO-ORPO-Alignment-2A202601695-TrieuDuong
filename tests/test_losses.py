"""Numeric tests for the DPO/ORPO objectives.

Expected values are recomputed here from the mathematical definition with plain
`math`, independently of the NumPy implementation under test -- so the tests pin
down the maths, not the code's own arithmetic.
"""
import math

import numpy as np
import pytest

from preference_lab.losses import dpo_loss, log1mexp, log_sigmoid, orpo_loss

LOG2 = math.log(2.0)


def ref_neg_log_sigmoid(x: float) -> float:
    """-log(sigmoid(x)) computed from the definition, for moderate x."""
    return math.log1p(math.exp(-x))


def ref_log_odds(logp: float) -> float:
    """log(p / (1 - p)) from a log probability."""
    return logp - math.log(1.0 - math.exp(logp))


# --------------------------------------------------------------------------- #
# Helper primitives
# --------------------------------------------------------------------------- #

def test_log_sigmoid_matches_definition_in_the_easy_range() -> None:
    xs = np.array([-5.0, -1.0, -0.06, 0.0, 0.5, 3.0, 12.0])
    expected = [-ref_neg_log_sigmoid(float(x)) for x in xs]
    assert log_sigmoid(xs) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_log_sigmoid_is_stable_where_the_naive_form_overflows() -> None:
    x = np.array([-800.0, -1e6])
    with np.errstate(over="ignore", divide="ignore"):
        naive = np.log(1.0 / (1.0 + np.exp(-x)))
    assert not np.all(np.isfinite(naive)), "naive form is expected to blow up here"
    stable = log_sigmoid(x)
    assert np.all(np.isfinite(stable))
    # For very negative x, log(sigmoid(x)) -> x.
    assert stable == pytest.approx(x, rel=1e-12)


def test_log_sigmoid_of_zero_is_minus_log_two() -> None:
    assert float(log_sigmoid(np.array([0.0]))[0]) == pytest.approx(-LOG2)


def test_log1mexp_matches_definition_on_both_branches() -> None:
    a = np.array([-0.1, -0.5, -LOG2, -1.0, -5.0, -40.0])
    expected = [math.log(1.0 - math.exp(float(v))) for v in a]
    assert log1mexp(a) == pytest.approx(expected, rel=1e-10, abs=1e-12)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #

def test_dpo_loss_known_value() -> None:
    # policy log-ratio 1.0, reference log-ratio 0.4 -> margin = beta * 0.6 = 0.06
    loss = dpo_loss(
        np.array([-0.5]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1
    )
    assert loss == pytest.approx(ref_neg_log_sigmoid(0.06), rel=1e-12)


def test_dpo_loss_is_log_two_when_policy_matches_reference() -> None:
    """No improvement over the reference model -> zero margin -> loss = log 2."""
    logps = np.array([-0.5, -2.0])
    rejected = np.array([-1.5, -3.0])
    loss = dpo_loss(logps, rejected, logps, rejected, beta=0.1)
    assert loss == pytest.approx(LOG2, rel=1e-12)


def test_dpo_loss_averages_over_the_batch() -> None:
    loss = dpo_loss(
        np.array([-0.5, -0.5]),
        np.array([-1.5, -2.5]),
        np.array([-0.6, -0.6]),
        np.array([-1.0, -1.0]),
        beta=0.1,
    )
    expected = (ref_neg_log_sigmoid(0.06) + ref_neg_log_sigmoid(0.1 * (2.0 - 0.4))) / 2.0
    assert loss == pytest.approx(expected, rel=1e-12)


def test_dpo_loss_decreases_as_the_margin_grows() -> None:
    losses = [
        dpo_loss(
            np.array([-0.5]), np.array([-gap]), np.array([-0.6]), np.array([-1.0]), beta=0.1
        )
        for gap in (0.6, 1.5, 4.0, 10.0)
    ]
    assert losses == sorted(losses, reverse=True)
    assert all(value > 0.0 for value in losses)


def test_dpo_loss_penalizes_a_reversed_preference() -> None:
    """Ranking the rejected answer higher must cost more than log 2."""
    loss = dpo_loss(
        np.array([-2.0]), np.array([-0.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1
    )
    assert loss > LOG2


def test_dpo_larger_beta_sharpens_the_penalty() -> None:
    args = (np.array([-2.0]), np.array([-0.5]), np.array([-0.6]), np.array([-1.0]))
    assert dpo_loss(*args, beta=1.0) > dpo_loss(*args, beta=0.1)


def test_dpo_loss_is_finite_for_extreme_logprobs() -> None:
    loss = dpo_loss(
        np.array([-1e6]), np.array([-1.0]), np.array([-1.0]), np.array([-1.0]), beta=1.0
    )
    assert math.isfinite(loss)
    assert loss == pytest.approx(1e6 - 1.0, rel=1e-9)


@pytest.mark.parametrize("beta", [0.0, -0.1, float("nan"), float("inf")])
def test_dpo_rejects_bad_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(np.array([-0.5]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=beta)


def test_dpo_rejects_batch_size_mismatch() -> None:
    with pytest.raises(ValueError, match="batch size mismatch"):
        dpo_loss(
            np.array([-0.5, -0.5]),
            np.array([-1.5]),
            np.array([-0.6]),
            np.array([-1.0]),
            beta=0.1,
        )


def test_dpo_rejects_positive_logprobs() -> None:
    """Passing raw logits instead of log-probs is a silent, costly bug."""
    with pytest.raises(ValueError, match="log probabilities"):
        dpo_loss(
            np.array([2.5]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1
        )


def test_dpo_rejects_non_finite_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        dpo_loss(
            np.array([-np.inf]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1
        )
    with pytest.raises(ValueError, match="empty"):
        dpo_loss(np.array([]), np.array([]), np.array([]), np.array([]), beta=0.1)


# --------------------------------------------------------------------------- #
# ORPO
# --------------------------------------------------------------------------- #

def test_orpo_loss_known_value() -> None:
    loss = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    log_odds_gap = ref_log_odds(-0.5) - ref_log_odds(-1.5)
    expected = 1.0 + 0.1 * ref_neg_log_sigmoid(log_odds_gap)
    assert loss == pytest.approx(expected, rel=1e-10)


def test_orpo_with_zero_lambda_is_pure_sft() -> None:
    nll = np.array([1.0, 3.0])
    loss = orpo_loss(nll, np.array([-0.5, -0.2]), np.array([-1.5, -4.0]), lambda_orpo=0.0)
    assert loss == pytest.approx(float(np.mean(nll)), rel=1e-12)


def test_orpo_odds_term_vanishes_when_preference_is_obvious() -> None:
    """chosen almost certain, rejected almost impossible -> penalty ~ 0."""
    loss = orpo_loss(np.array([0.0]), np.array([-1e-6]), np.array([-60.0]), lambda_orpo=1.0)
    assert loss == pytest.approx(0.0, abs=1e-9)


def test_orpo_penalizes_a_reversed_preference() -> None:
    good = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=1.0)
    bad = orpo_loss(np.array([1.0]), np.array([-1.5]), np.array([-0.5]), lambda_orpo=1.0)
    assert bad > good


def test_orpo_larger_lambda_weights_the_preference_term_more() -> None:
    args = (np.array([1.0]), np.array([-1.5]), np.array([-0.5]))
    assert orpo_loss(*args, lambda_orpo=1.0) > orpo_loss(*args, lambda_orpo=0.1)


def test_orpo_is_finite_at_the_probability_one_boundary() -> None:
    """logp == 0 means p == 1, where the odds ratio diverges; clamping must save us."""
    loss = orpo_loss(np.array([0.0]), np.array([0.0]), np.array([-1.0]), lambda_orpo=0.1)
    assert math.isfinite(loss)


def test_orpo_is_finite_for_extreme_logprobs() -> None:
    loss = orpo_loss(np.array([5.0]), np.array([-700.0]), np.array([-1e-8]), lambda_orpo=0.1)
    assert math.isfinite(loss)


def test_orpo_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="lambda_orpo"):
        orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=-1.0)
    with pytest.raises(ValueError, match="sft_nll must be non-negative"):
        orpo_loss(np.array([-1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    with pytest.raises(ValueError, match="log probabilities"):
        orpo_loss(np.array([1.0]), np.array([0.5]), np.array([-1.5]), lambda_orpo=0.1)
    with pytest.raises(ValueError, match="batch size mismatch"):
        orpo_loss(np.array([1.0, 1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
