#!/usr/bin/env python3
"""Plot MedMCQA accuracy before and after a LoRA fine-tune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_BASELINE_METRICS = Path("outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/metrics.json")
DEFAULT_ADAPTER_METRICS = Path("artifacts/adapter_evals/grpo_10k_ft_leftpad/medmcqa/metrics.json")
DEFAULT_OUTPUT = Path("artifacts/adapter_evals/grpo_10k_ft_leftpad/medmcqa/generation_accuracy_comparison.png")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Instruct vs fine-tuned MedMCQA accuracy.")
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--adapter-metrics", type=Path, default=DEFAULT_ADAPTER_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_frame(baseline_metrics_path: Path, adapter_metrics_path: Path) -> pd.DataFrame:
    baseline_payload = load_json(baseline_metrics_path)
    adapter_payload = load_json(adapter_metrics_path)
    baseline = baseline_payload.get("baseline", baseline_payload)

    rows = [
        {
            "model": "OLMo-2-1B-Instruct",
            "metric": "Generated",
            "accuracy": float(baseline["generated_accuracy"]),
        },
        {
            "model": "GRPO MedMCQA Fine-Tune",
            "metric": "Generated",
            "accuracy": float(adapter_payload["generated_accuracy"]),
        },
    ]
    return pd.DataFrame(rows)


def plot_accuracy(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(
        frame["model"],
        frame["accuracy"],
        color=["#4C78A8", "#59A14F"],
        width=0.56,
    )
    for bar, value in zip(bars, frame["accuracy"]):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.20",
        )

    ax.set_title("MedMCQA Generation Accuracy Before And After GRPO Fine-Tuning")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, max(0.42, float(frame["accuracy"].max()) + 0.05))
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    frame = build_frame(args.baseline_metrics, args.adapter_metrics)
    plot_accuracy(frame, args.output)
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_output, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
