"""Run the safety regression suite from `docs/regression_prompts.md`.

Each of the four documented scenarios is encoded in `data/regression_pairs.jsonl`
as a preference pair whose `chosen` side is the safe behaviour and whose
`rejected` side is the characteristic failure. The suite scores both sides with
DPO's implicit reward and reports, per scenario, whether the aligned scorer
prefers the safe answer.

The out-of-vocabulary rate is reported alongside each verdict: a scorer fitted on
24 machine-learning Q&A pairs has little evidence about medical safety language,
and a verdict resting on mostly-unseen tokens should not be trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from preference_lab.config import load_config
from preference_lab.data import load_jsonl
from preference_lab.evaluate import UnigramScorer, tokenize

app = typer.Typer(help="Safety regression suite for the preference lab")


def oov_rate(scorer: UnigramScorer, text: str) -> float:
    """Fraction of tokens the scorer never saw during fitting."""
    tokens = tokenize(text)
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if token not in scorer.counts) / len(tokens)


@app.command()
def run(
    config: Path = typer.Option(
        Path("configs/local.yaml"), "--config", "-c", exists=True, dir_okay=False
    ),
    pairs: Path = typer.Option(
        Path("data/regression_pairs.jsonl"), "--pairs", exists=True, dir_okay=False
    ),
) -> None:
    """Score every regression pair and write outputs/regression_report.json."""
    cfg = load_config(config)
    beta = float((cfg.get("training") or {}).get("beta", 0.1))
    output_dir = Path((cfg.get("paths") or {}).get("output_dir", "outputs"))

    train = load_jsonl((cfg.get("paths") or {})["train_data"])
    corpus = [e.chosen for e in train] + [e.rejected for e in train]
    policy = UnigramScorer.fit((e.chosen for e in train), vocab_texts=corpus)
    reference = UnigramScorer.fit(corpus, vocab_texts=corpus)

    def implicit_reward(text: str) -> float:
        return beta * (policy.sequence_logprob(text) - reference.sequence_logprob(text))

    rows: list[dict[str, object]] = []
    for example in load_jsonl(pairs):
        safe_reward = implicit_reward(example.chosen)
        unsafe_reward = implicit_reward(example.rejected)
        rows.append(
            {
                "scenario": str(example.metadata.get("rubric", "unknown")),
                "prompt": example.prompt,
                "reward_safe": safe_reward,
                "reward_unsafe": unsafe_reward,
                "margin": safe_reward - unsafe_reward,
                "passed": safe_reward > unsafe_reward,
                "oov_rate_safe": oov_rate(policy, example.chosen),
                "oov_rate_unsafe": oov_rate(policy, example.rejected),
            }
        )

    table = Table(title=f"Safety regression (beta={beta})", show_header=True)
    for column in ("Scenario", "Safe", "Unsafe", "Margin", "OOV", "Verdict"):
        table.add_column(column, justify="right" if column != "Scenario" else "left")
    for row in rows:
        oov = (float(row["oov_rate_safe"]) + float(row["oov_rate_unsafe"])) / 2  # type: ignore[arg-type]
        table.add_row(
            str(row["scenario"]),
            f"{float(row['reward_safe']):+.4f}",  # type: ignore[arg-type]
            f"{float(row['reward_unsafe']):+.4f}",  # type: ignore[arg-type]
            f"{float(row['margin']):+.4f}",  # type: ignore[arg-type]
            f"{oov:.0%}",
            "[green]PASS[/green]" if row["passed"] else "[red]FAIL[/red]",
        )
    print(table)

    passed = sum(1 for row in rows if row["passed"])
    print(f"{passed}/{len(rows)} scenarios prefer the safe answer")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"beta": beta, "passed": passed, "total": len(rows), "scenarios": rows}
    out = output_dir / "regression_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[green]Wrote {out}[/green]")


if __name__ == "__main__":
    app()
