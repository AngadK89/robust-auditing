#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot unique SFT examples seen versus ProFLingo TRR.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, default=None)
    parser.add_argument("--csv-name", default="sft_examples_vs_proflingo.csv")
    parser.add_argument("--png-name", default="sft_examples_vs_proflingo.png")
    return parser


def write_plot_artifacts(
    *,
    results_path: Path,
    output_dir: Path,
    schedule_path: Path | None = None,
    csv_name: str = "sft_examples_vs_proflingo.csv",
    png_name: str = "sft_examples_vs_proflingo.png",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_by_step = load_schedule(schedule_path) if schedule_path else {}
    rows = normalize_results(read_jsonl(results_path), schedule_by_step=schedule_by_step)
    csv_path = output_dir / csv_name
    png_path = output_dir / png_name
    write_csv(csv_path, rows)
    write_png(png_path, rows)
    summary = {
        "results_path": str(results_path),
        "schedule_path": str(schedule_path) if schedule_path else None,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "points": rows,
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def normalize_results(
    results: Sequence[Mapping[str, Any]],
    *,
    schedule_by_step: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        step = int(result.get("step") or 0)
        scheduled = schedule_by_step.get(step, {})
        unique_seen = result.get("unique_source_examples_seen", scheduled.get("unique_source_examples_seen"))
        matched = int(result.get("matched") or 0)
        total = int(result.get("total") or 0)
        trr = float(result.get("trr", result.get("match_rate", (matched / total) if total else 0.0)))
        rows.append(
            {
                "step": step,
                "unique_source_examples_seen": int(unique_seen or 0),
                "matched": matched,
                "total": total,
                "trr": trr,
                "adapter_dir": result.get("adapter_dir"),
                "checkpoint_kind": result.get("checkpoint_kind"),
            }
        )
    return sorted(rows, key=lambda row: (row["unique_source_examples_seen"], row["step"]))


def load_schedule(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["step"]): row for row in read_jsonl(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "step",
        "unique_source_examples_seen",
        "matched",
        "total",
        "trr",
        "checkpoint_kind",
        "adapter_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_png(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    xs = [int(row["unique_source_examples_seen"]) for row in rows]
    ys = [float(row["trr"]) for row in rows]
    labels = [f"{float(row['trr']):.2f}" for row in rows]

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 240
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    line_marker_size = 8
    line_width = 2.8
    score_label_font_size = 10
    grid_color = "#E5E7EB"

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, ys, marker="o", markersize=line_marker_size, linewidth=line_width, color="#7A5195")
    annotated_positions: set[tuple[int, float]] = set()
    for x, y, label in zip(xs, ys, labels):
        position = (x, y)
        if position in annotated_positions:
            continue
        annotated_positions.add(position)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=score_label_font_size,
            color="#1F2937",
        )
    ax.set_xlabel("Unique SFT Examples Observed")
    ax.set_ylabel("ProFLingo Target Response Rate (TRR)")
    ax.set_title("SFT Training Examples vs. ProFLingo Fingerprint Retention", pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color=grid_color)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)




def main() -> int:
    args = build_parser().parse_args()
    summary = write_plot_artifacts(
        results_path=args.results,
        output_dir=args.output_dir,
        schedule_path=args.schedule,
        csv_name=args.csv_name,
        png_name=args.png_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
