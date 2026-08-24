from __future__ import annotations

import json
import random
import re
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from .schemas import PreferenceExample, normalize_text

#: Coarse PII patterns. Deliberately conservative: this is a guardrail that flags
#: suspicious records for a human, not a compliance-grade PII detector.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
}

PiiAction = Literal["ignore", "warn", "raise"]


def scan_pii(text: str) -> list[str]:
    """Return the names of PII patterns that match ``text``."""
    return [name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)]


def load_jsonl(
    path: str | Path,
    *,
    allow_duplicate_prompts: bool = False,
    pii_action: PiiAction = "warn",
) -> list[PreferenceExample]:
    """Load preference examples from JSONL with line-numbered diagnostics.

    Args:
        path: Path to the JSONL file. Blank lines are skipped.
        allow_duplicate_prompts: When False (default), a repeated prompt raises.
            Duplicate prompts inflate evaluation scores and are a classic sign of a
            dirty corpus, so failing loudly is the safer default.
        pii_action: What to do when a record trips a PII pattern -- ``"ignore"``,
            ``"warn"`` (default), or ``"raise"``.

    Returns:
        The parsed examples, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On malformed JSON, schema violations, duplicate prompts, or --
            when ``pii_action="raise"`` -- suspected PII. Every message is prefixed
            with ``<path>:<line_no>:`` so the offending record is easy to find.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Preference dataset not found: {file_path}")

    examples: list[PreferenceExample] = []
    seen_prompts: dict[str, int] = {}

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                payload: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_no}: invalid JSON - {exc}") from exc

            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{file_path}:{line_no}: invalid schema - {exc}") from exc

            if not allow_duplicate_prompts:
                key = normalize_text(example.prompt)
                first_seen = seen_prompts.get(key)
                if first_seen is not None:
                    raise ValueError(
                        f"{file_path}:{line_no}: duplicate prompt, first seen on line "
                        f"{first_seen}: {example.prompt[:80]!r}"
                    )
                seen_prompts[key] = line_no

            if pii_action != "ignore":
                _enforce_pii(example, file_path, line_no, pii_action)

            examples.append(example)

    return examples


def _enforce_pii(
    example: PreferenceExample,
    file_path: Path,
    line_no: int,
    pii_action: PiiAction,
) -> None:
    """Apply the configured PII policy to a single record."""
    hits = sorted(
        set(scan_pii(example.prompt) + scan_pii(example.chosen) + scan_pii(example.rejected))
    )
    if not hits:
        return
    message = f"{file_path}:{line_no}: possible PII detected ({', '.join(hits)})"
    if pii_action == "raise":
        raise ValueError(message)
    warnings.warn(message, stacklevel=3)


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Splitting on row index lets two rows that share a prompt land on opposite sides
    of the split, so the validation score measures memorization rather than
    generalization. Grouping by prompt first makes that impossible.

    The prompt order is shuffled with ``random.Random(seed)`` so the split is
    reproducible across runs and machines.

    Args:
        examples: Examples to split.
        validation_ratio: Target fraction of *prompt groups* held out, in ``[0, 1]``.
        seed: Seed for the deterministic shuffle.

    Returns:
        ``(train, validation)``. Every input example appears in exactly one side, so
        ``len(train) + len(validation) == len(examples)`` always holds.

    Raises:
        ValueError: If ``validation_ratio`` is outside ``[0, 1]``.
    """
    if not 0.0 <= validation_ratio <= 1.0:
        raise ValueError(f"validation_ratio must be within [0, 1], got {validation_ratio}")
    if not examples:
        return [], []

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(normalize_text(example.prompt), []).append(example)

    prompt_keys = list(groups)
    random.Random(seed).shuffle(prompt_keys)

    n_prompts = len(prompt_keys)
    n_train = round(n_prompts * (1.0 - validation_ratio))
    # Keep at least one group on each side whenever there is more than one group,
    # so neither split is silently empty.
    if n_prompts > 1:
        n_train = min(max(n_train, 1), n_prompts - 1)

    train: list[PreferenceExample] = []
    validation: list[PreferenceExample] = []
    for index, key in enumerate(prompt_keys):
        target = train if index < n_train else validation
        target.extend(groups[key])

    return train, validation
