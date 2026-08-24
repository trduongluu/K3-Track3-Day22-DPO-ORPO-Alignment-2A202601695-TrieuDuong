# Preference Alignment Lab: DPO \& ORPO Starter

Production-style skeleton for a 2-hour lab on preference alignment. The repository is intentionally incomplete: students must implement the logic marked `TODO(student)`.

## Learning goals

- Validate and load preference pairs (`prompt`, `chosen`, `rejected`).
- Implement or wrap DPO/ORPO training logic.
- Build evaluation metrics for pairwise preference and regression prompts.
- Practice production habits: typed code, configs, tests, Makefile, CI, docs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
```

Optional training dependencies:

```bash
pip install -e '.[dev,train]'
```

## Lab rules

1. Do not rewrite the whole repository.
2. Implement only the `TODO(student)` blocks unless you have a clear reason.
3. Keep tests passing after each milestone.
4. Do not commit secrets, model weights, or private datasets.

## Milestones

| Time | Goal | Command |
|---|---|---|
| 0-30 min | Setup and inspect sample data | `make test` |
| 30-50 min | Implement dataset validation/collator | `pytest tests/test_data.py` |
| 50-70 min | (Optional) Generate synthetic data | `python scripts/generate_data.py` |
| 70-100 min | Implement DPO or ORPO TODO | `pytest tests/test_losses.py` |
| 100-115 min | Implement evaluation and report | `pref-lab evaluate --config configs/local.yaml` |
| 115-120 min | One-minute demo | `cat outputs/metrics.json` |

## Repository layout

```text
src/preference_lab/     Python package
data/                   Small sample preference dataset
configs/                YAML configs for local experiments
docs/                   Lab guide, rubric, data card template
scripts/                Utility entrypoints
tests/                  Unit tests for student work
```

## Production checklist

- [x] Dataset schema validated. — `PreferenceExample` rejects empty fields, identical
      `chosen`/`rejected` (case- and whitespace-insensitive) and near-duplicates; `load_jsonl`
      adds line-numbered errors, duplicate-prompt detection and a PII guardrail.
- [x] Train/eval split by prompt, not by row. — `split_by_prompt` groups by normalized prompt,
      shuffles prompt keys with `random.Random(seed)`, then cuts on a group boundary. Tests assert
      both `len(train) + len(val) == len(examples)` and an empty prompt intersection.
- [x] Config committed; generated artifacts ignored. — `configs/local.yaml` holds every knob;
      `.gitignore` keeps `outputs/` out of version control, so measured values are quoted in
      `docs/REPORT_TEMPLATE.md` instead.
- [x] Metrics saved as JSON. — `outputs/metrics.json`, plus `checkpoint.json`,
      `training_metrics.json` and `regression_report.json`.
- [x] Safety regression prompts run before/after training. — `python scripts/run_regression.py`
      scores the four scenarios with the un-aligned reference vs. the aligned policy: 3/4 prefer
      the safe answer. See the Safety section of the report for why the passes are weaker than
      they look.
- [x] Data card updated. — `docs/data_card_template.md`, including the measured length bias.

## Results

`make test` 85 passed · `make lint` clean · `make typecheck` clean (mypy `strict`) · no
`NotImplementedError` left in `src/`.

| Metric | Value |
|---|---|
| Pairwise accuracy (6 held-out prompts) | 83.33% |
| DPO loss | 0.5209 (vs. `log 2 = 0.6931` for an uninformative policy) |
| "Always prefer the longer answer" baseline | 100% — the corpus's built-in length bias |
| Safety regression | 3/4 scenarios prefer the safe answer |

Both objectives are implemented. Full write-up, including three observed failure modes and a
justification for every change made outside a `TODO(student)` block, is in
[docs/REPORT_TEMPLATE.md](docs/REPORT_TEMPLATE.md).

## Commands

```bash
pref-lab validate data/sample_preferences.jsonl   # -> Loaded 24 preference examples
pref-lab evaluate --config configs/local.yaml     # -> outputs/metrics.json
pref-lab train    --config configs/local.yaml     # -> outputs/checkpoint.json (CPU mock trainer)
python scripts/run_regression.py                  # -> outputs/regression_report.json
```
