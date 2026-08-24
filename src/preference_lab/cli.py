from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich import print
from rich.table import Table

from .config import load_config
from .data import load_jsonl
from .evaluate import evaluate_preferences, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a config section as a dict, tolerating a missing or null section."""
    value = cfg.get(name) or {}
    if not isinstance(value, dict):
        raise typer.BadParameter(f"config section '{name}' must be a mapping, got {type(value)}")
    return value


@app.command()
def validate(data: Path) -> None:
    """Load a preference dataset and report how many examples survived validation."""
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def evaluate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="YAML config, e.g. configs/local.yaml",
    ),
) -> None:
    """Score the held-out split with the deterministic scorer and write metrics.json.

    Declared as an option (not a positional argument) so the documented invocation
    ``pref-lab evaluate --config configs/local.yaml`` -- used by ``make run-eval``
    and ``scripts/smoke_test.sh`` -- actually works.
    """
    cfg = load_config(config)
    paths = _section(cfg, "paths")
    training = _section(cfg, "training")
    evaluation = _section(cfg, "evaluation")

    examples = load_jsonl(paths["train_data"])
    metrics = evaluate_preferences(
        examples,
        validation_ratio=float(evaluation.get("validation_ratio", 0.25)),
        seed=int(cfg.get("seed", 42)),
        beta=float(training.get("beta", 0.1)),
        lambda_orpo=float(training.get("lambda_orpo", 0.1)),
        tie_credit=float(evaluation.get("tie_credit", 0.5)),
        smoothing_alpha=float(evaluation.get("smoothing_alpha", 1.0)),
    )

    table = Table(title="Evaluation metrics", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in sorted(metrics):
        value = metrics[key]
        table.add_row(key, f"{value}" if isinstance(value, int) else f"{value:.6f}")
    print(table)

    out = write_metrics(metrics, paths["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


@app.command()
def train(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, dir_okay=False, help="YAML config"
    ),
) -> None:
    """Run the CPU mock trainer and write checkpoint.json + training_metrics.json."""
    trainer = PreferenceTrainer(TrainingConfig.from_mapping(load_config(config)))
    trainer.train()

    first, last = trainer.history[0], trainer.history[-1]
    table = Table(title=f"Training ({trainer.config.method})", show_header=True)
    table.add_column("Metric")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    for key in ("train_loss", "val_loss", "val_pairwise_accuracy"):
        table.add_row(key, f"{float(first[key]):.6f}", f"{float(last[key]):.6f}")
    print(table)
    print(
        "[dim]weights "
        + ", ".join(
            f"{name}={value:+.4f}"
            for name, value in zip(PreferenceTrainer.FEATURES, trainer.weights, strict=True)
        )
        + "[/dim]"
    )
    print(f"[green]Wrote checkpoint to {trainer.config.output_dir / 'checkpoint.json'}[/green]")


if __name__ == "__main__":
    app()
