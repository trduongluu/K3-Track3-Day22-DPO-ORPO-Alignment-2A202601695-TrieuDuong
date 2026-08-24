"""Numerically stable preference-alignment objectives (DPO and ORPO).

Both objectives are implemented in pure NumPy so the lab runs on CPU. They take
*sequence-level* log probabilities -- i.e. the sum of token log-probs for a whole
completion -- which is exactly what a TRL-backed trainer would hand them.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Log probabilities are clamped just below 0 before any odds computation, because
#: ``odds = p / (1 - p)`` diverges as ``p -> 1``.
_LOGP_EPS = 1e-9


def _as_float_array(name: str, value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Coerce to a 1-D float64 array and reject non-finite entries."""
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values (nan/inf)")
    return array


def _require_same_shape(**arrays: npt.NDArray[np.float64]) -> None:
    sizes = {name: array.size for name, array in arrays.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"batch size mismatch: {sizes}")


def _require_log_probs(name: str, array: npt.NDArray[np.float64]) -> None:
    """Guard against the classic bug of passing raw logits instead of log-probs."""
    if np.any(array > 0.0):
        raise ValueError(f"{name} must contain log probabilities (<= 0), got max {array.max()}")


def log_sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute ``log(sigmoid(x))`` without overflow.

    The naive form ``log(1 / (1 + exp(-x)))`` overflows to ``-inf``/``nan`` once
    ``x`` drops below roughly ``-745``. Rewriting it as ``-softplus(-x)`` and using
    the stable softplus identity ``softplus(z) = max(z, 0) + log1p(exp(-|z|))``
    keeps every intermediate value bounded.
    """
    z = -x
    result: npt.NDArray[np.float64] = -(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))))
    return result


def log1mexp(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute ``log(1 - exp(a))`` for ``a < 0`` without catastrophic cancellation.

    Uses Maechler's two-branch trick: near ``a = 0`` the accurate form is
    ``log(-expm1(a))``; far from it, ``log1p(-exp(a))``. Splitting at ``-log(2)``
    keeps the relative error small on both sides.
    """
    cutoff = -np.log(2.0)
    safe = np.minimum(a, -_LOGP_EPS)
    result: npt.NDArray[np.float64] = np.where(
        safe > cutoff, np.log(-np.expm1(safe)), np.log1p(-np.exp(safe))
    )
    return result


def dpo_loss(
    policy_chosen_logps: npt.NDArray[np.float64],
    policy_rejected_logps: npt.NDArray[np.float64],
    ref_chosen_logps: npt.NDArray[np.float64],
    ref_rejected_logps: npt.NDArray[np.float64],
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    The objective is::

        L = -mean( log sigmoid( beta * [ (pi_c - pi_r) - (ref_c - ref_r) ] ) )

    The bracketed term is the *implicit reward margin*: how much further apart the
    policy pushes chosen and rejected than the frozen reference model already did.
    Subtracting the reference log-ratio is what keeps the policy anchored to natural
    language instead of drifting to whatever degenerate text maximizes the margin.

    Args:
        policy_chosen_logps: Policy log-probs of the chosen completions.
        policy_rejected_logps: Policy log-probs of the rejected completions.
        ref_chosen_logps: Reference-model log-probs of the chosen completions.
        ref_rejected_logps: Reference-model log-probs of the rejected completions.
        beta: Temperature > 0. Larger beta trusts the preference signal less and
            stays closer to the reference model.

    Returns:
        Mean loss over the batch. Always >= 0, and -> 0 as the margin -> +inf.

    Raises:
        ValueError: On empty/non-finite inputs, batch-size mismatch, positive
            log-probs, or non-positive ``beta``.
    """
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be a positive finite float, got {beta}")

    pi_chosen = _as_float_array("policy_chosen_logps", policy_chosen_logps)
    pi_rejected = _as_float_array("policy_rejected_logps", policy_rejected_logps)
    ref_chosen = _as_float_array("ref_chosen_logps", ref_chosen_logps)
    ref_rejected = _as_float_array("ref_rejected_logps", ref_rejected_logps)
    _require_same_shape(
        policy_chosen_logps=pi_chosen,
        policy_rejected_logps=pi_rejected,
        ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected,
    )
    for name, array in (
        ("policy_chosen_logps", pi_chosen),
        ("policy_rejected_logps", pi_rejected),
        ("ref_chosen_logps", ref_chosen),
        ("ref_rejected_logps", ref_rejected),
    ):
        _require_log_probs(name, array)

    policy_logratios = pi_chosen - pi_rejected
    ref_logratios = ref_chosen - ref_rejected
    margins = beta * (policy_logratios - ref_logratios)
    return float(-np.mean(log_sigmoid(margins)))


def orpo_loss(
    sft_nll: npt.NDArray[np.float64],
    chosen_logps: npt.NDArray[np.float64],
    rejected_logps: npt.NDArray[np.float64],
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    The objective is::

        L = mean(sft_nll) + lambda * mean( -log sigmoid( log_odds_c - log_odds_r ) )

    where ``log_odds(y) = logp(y) - log(1 - exp(logp(y)))``.

    ORPO needs no reference model: the SFT term alone anchors the policy to natural
    language, and the odds-ratio term supplies the preference signal. That halves
    the memory footprint compared with DPO, which must keep a frozen copy resident.

    Args:
        sft_nll: Per-example supervised negative log-likelihood of the chosen
            completion (>= 0).
        chosen_logps: Log-probs of the chosen completions (<= 0).
        rejected_logps: Log-probs of the rejected completions (<= 0).
        lambda_orpo: Weight of the odds-ratio penalty. Must be >= 0.

    Returns:
        Mean loss over the batch.

    Raises:
        ValueError: On empty/non-finite inputs, batch-size mismatch, positive
            log-probs, negative ``sft_nll``, or negative ``lambda_orpo``.
    """
    if not np.isfinite(lambda_orpo) or lambda_orpo < 0.0:
        raise ValueError(f"lambda_orpo must be a non-negative finite float, got {lambda_orpo}")

    nll = _as_float_array("sft_nll", sft_nll)
    chosen = _as_float_array("chosen_logps", chosen_logps)
    rejected = _as_float_array("rejected_logps", rejected_logps)
    _require_same_shape(sft_nll=nll, chosen_logps=chosen, rejected_logps=rejected)
    if np.any(nll < 0.0):
        raise ValueError(f"sft_nll must be non-negative, got min {nll.min()}")
    _require_log_probs("chosen_logps", chosen)
    _require_log_probs("rejected_logps", rejected)

    # Clamp strictly below 0 so `1 - exp(logp)` never collapses to exactly 0.
    safe_chosen = np.minimum(chosen, -_LOGP_EPS)
    safe_rejected = np.minimum(rejected, -_LOGP_EPS)

    log_odds_chosen = safe_chosen - log1mexp(safe_chosen)
    log_odds_rejected = safe_rejected - log1mexp(safe_rejected)

    odds_ratio_loss = -log_sigmoid(log_odds_chosen - log_odds_rejected)
    return float(np.mean(nll) + lambda_orpo * np.mean(odds_ratio_loss))
