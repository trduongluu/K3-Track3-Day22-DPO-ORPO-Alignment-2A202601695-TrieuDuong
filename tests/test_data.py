from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample

SAMPLE = "data/sample_preferences.jsonl"


def _write(tmp_path: Path, name: str, body: str) -> Path:
    target = tmp_path / name
    target.write_text(body, encoding="utf-8")
    return target


def test_load_sample_data() -> None:
    examples = load_jsonl(SAMPLE)
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_load_sample_data_has_no_duplicate_prompts() -> None:
    examples = load_jsonl(SAMPLE)
    assert len({e.prompt for e in examples}) == len(examples)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "gaps.jsonl",
        '{"prompt":"a","chosen":"b","rejected":"c"}\n\n\n{"prompt":"d","chosen":"e","rejected":"f"}\n',
    )
    assert len(load_jsonl(path)) == 2


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_jsonl("data/does_not_exist.jsonl")


def test_error_message_includes_line_number(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "bad.jsonl",
        '{"prompt":"a","chosen":"b","rejected":"c"}\n{oops\n',
    )
    with pytest.raises(ValueError, match="2"):
        load_jsonl(path)


def test_schema_error_message_includes_line_number(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "schema.jsonl",
        '{"prompt":"a","chosen":"b","rejected":"c"}\n{"prompt":"a2","chosen":"b2"}\n',
    )
    with pytest.raises(ValueError, match=r":2: invalid schema"):
        load_jsonl(path)


def test_duplicate_prompt_is_rejected_and_reports_first_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "dupes.jsonl",
        '{"prompt":"Same","chosen":"b","rejected":"c"}\n'
        '{"prompt":"  same  ","chosen":"x","rejected":"y"}\n',
    )
    with pytest.raises(ValueError, match="duplicate prompt, first seen on line 1"):
        load_jsonl(path)

    # ...but the check is opt-out for corpora that legitimately repeat prompts.
    assert len(load_jsonl(path, allow_duplicate_prompts=True)) == 2


def test_pii_guardrail_can_raise(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "pii.jsonl",
        '{"prompt":"Contact me","chosen":"Email trieu@example.com","rejected":"No"}\n',
    )
    with pytest.raises(ValueError, match="possible PII detected"):
        load_jsonl(path, pii_action="raise")
    assert len(load_jsonl(path, pii_action="ignore")) == 1


def test_chosen_and_rejected_must_differ_ignoring_case_and_whitespace() -> None:
    with pytest.raises(ValueError, match="must differ"):
        PreferenceExample(prompt="p", chosen="Yes  It   Works", rejected="yes it works")


def test_near_duplicate_pair_is_rejected() -> None:
    base = "Self-attention weighs every token against every other token in the sequence."
    # Differs by a single character: not equal, but carries no preference signal.
    with pytest.raises(ValueError, match="near-duplicates"):
        PreferenceExample(prompt="p", chosen=base, rejected=base.replace("sequence.", "sequence!"))


def test_genuinely_different_pair_is_accepted() -> None:
    example = PreferenceExample(
        prompt="p",
        chosen="Self-attention weighs every token against every other token.",
        rejected="Self-attention is just a faster RNN with less memory.",
    )
    assert example.chosen != example.rejected


def test_split_returns_all_examples() -> None:
    examples = load_jsonl(SAMPLE)
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_split_has_no_prompt_leakage() -> None:
    examples = load_jsonl(SAMPLE)
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)
    assert not ({e.prompt for e in train} & {e.prompt for e in val})


def test_split_groups_rows_sharing_a_prompt() -> None:
    shared = [
        PreferenceExample(prompt="shared", chosen=f"good {i}", rejected=f"bad {i}")
        for i in range(6)
    ]
    others = [
        PreferenceExample(prompt=f"other {i}", chosen="good", rejected="bad") for i in range(6)
    ]
    train, val = split_by_prompt(shared + others, validation_ratio=0.5)
    assert len(train) + len(val) == 12
    # All six rows of the shared prompt must land on the same side.
    assert not ({e.prompt for e in train} & {e.prompt for e in val})


def test_split_is_deterministic_for_a_seed() -> None:
    examples = load_jsonl(SAMPLE)
    first = split_by_prompt(examples, validation_ratio=0.25, seed=42)
    second = split_by_prompt(examples, validation_ratio=0.25, seed=42)
    assert [e.prompt for e in first[0]] == [e.prompt for e in second[0]]
    assert [e.prompt for e in first[1]] == [e.prompt for e in second[1]]


def test_split_seed_changes_the_partition() -> None:
    examples = load_jsonl(SAMPLE)
    val_a = {e.prompt for e in split_by_prompt(examples, validation_ratio=0.25, seed=42)[1]}
    val_b = {e.prompt for e in split_by_prompt(examples, validation_ratio=0.25, seed=7)[1]}
    assert val_a != val_b


def test_split_never_empties_a_side() -> None:
    examples = load_jsonl(SAMPLE)
    train, val = split_by_prompt(examples, validation_ratio=0.0)
    assert len(val) >= 1
    train, val = split_by_prompt(examples, validation_ratio=1.0)
    assert len(train) >= 1


def test_split_of_empty_input() -> None:
    assert split_by_prompt([], validation_ratio=0.5) == ([], [])


def test_split_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError, match="validation_ratio"):
        split_by_prompt(load_jsonl(SAMPLE), validation_ratio=1.5)
