from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

_WHITESPACE_RE = re.compile(r"\s+")

#: Normalized similarity above this ratio is treated as a near-duplicate pair.
NEAR_DUPLICATE_THRESHOLD = 0.98


def normalize_text(value: str) -> str:
    """Normalize text for comparison: casefold and collapse all whitespace runs.

    Comparing raw strings is too permissive: ``"Yes"`` and ``"yes  "`` are the same
    answer for preference-learning purposes, so a pair built from them carries no
    usable training signal.
    """
    return _WHITESPACE_RE.sub(" ", value).strip().casefold()


def similarity(left: str, right: str) -> float:
    """Return a 0..1 similarity ratio between two already-normalized strings."""
    return SequenceMatcher(None, left, right).ratio()


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: ValidationInfo) -> str:
        chosen = info.data.get("chosen")
        if not isinstance(chosen, str):
            # `chosen` failed its own validation; pydantic already reports that error.
            return rejected

        normalized_chosen = normalize_text(chosen)
        normalized_rejected = normalize_text(rejected)

        if normalized_chosen == normalized_rejected:
            raise ValueError("chosen and rejected must differ (ignoring case/whitespace)")

        ratio = similarity(normalized_chosen, normalized_rejected)
        if ratio >= NEAR_DUPLICATE_THRESHOLD:
            raise ValueError(
                f"chosen and rejected are near-duplicates (similarity={ratio:.4f} "
                f">= {NEAR_DUPLICATE_THRESHOLD}); the pair carries no preference signal"
            )
        return rejected
