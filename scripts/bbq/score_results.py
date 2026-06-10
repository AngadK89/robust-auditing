from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from robust_auditing.bbq.artifacts import read_jsonl, write_csv, write_json
from robust_auditing.bbq.formatting import SUPPORTED_FORMATS
from robust_auditing.bbq.metrics import BBQScoreResult, score_prediction_rows
from robust_auditing.bbq.targets import DEFAULT_TARGET_IDS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score BBQ prediction artifacts.")
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/bbq/runs/olmo2_instruct_clean_exact_chain_10k_seed0"))
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGET_IDS))
    parser.add_argument("--formats", nargs="+", default=list(SUPPORTED_FORMATS), choices=SUPPORTED_FORMATS)
    return parser


def score_run(*, run_dir: Path, targets: Sequence[str], formats: Sequence[str]) -> dict[str, Any]:
    comparison_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for target_id in targets:
        for prompt_format in formats:
            prediction_path = run_dir / "predictions" / target_id / prompt_format / "predictions.jsonl"
            if not prediction_path.exists():
                continue
            result = score_prediction_rows(read_jsonl(prediction_path))
            metric_dir = run_dir / "metrics" / target_id / prompt_format
            write_score_result(metric_dir, result)
            summaries[f"{target_id}/{prompt_format}"] = result.summary
            comparison_rows.extend(_comparison_rows(result, target_id=target_id, prompt_format=prompt_format))
    write_csv(run_dir / "comparison" / "summary.csv", comparison_rows)
    return {"summaries": summaries, "comparison_rows": len(comparison_rows)}


def write_score_result(metric_dir: Path, result: BBQScoreResult) -> None:
    write_json(metric_dir / "summary.json", result.summary)
    write_csv(metric_dir / "accuracy_by_category_context.csv", result.accuracy_by_category_context)
    write_csv(metric_dir / "bias_by_category_context.csv", result.bias_by_category_context)
    write_csv(metric_dir / "disambig_accuracy_alignment.csv", result.disambig_accuracy_alignment)


def _comparison_rows(result: BBQScoreResult, *, target_id: str, prompt_format: str) -> list[dict[str, Any]]:
    accuracy_by_key = {
        (row["category"], row["context_condition"]): row
        for row in result.accuracy_by_category_context
    }
    output: list[dict[str, Any]] = []
    for bias_row in result.bias_by_category_context:
        accuracy_row = accuracy_by_key.get((bias_row["category"], bias_row["context_condition"]), {})
        output.append(
            {
                "target_id": target_id,
                "format": prompt_format,
                "category": bias_row["category"],
                "context_condition": bias_row["context_condition"],
                "accuracy": accuracy_row.get("accuracy", bias_row.get("accuracy")),
                "bias_score_pct": bias_row["bias_score_pct"],
                "matched_count": accuracy_row.get("matched_count"),
                "n_non_unknown": bias_row["n_non_unknown"],
                "unmatched_rate": result.summary["unmatched_rate"],
            }
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = score_run(run_dir=args.run_dir, targets=tuple(args.targets), formats=tuple(args.formats))
    print(f"Wrote BBQ metrics for {len(summary['summaries'])} target/format pairs to {args.run_dir / 'metrics'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
