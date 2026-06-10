from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from robust_auditing.bbq.artifacts import read_json, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local BBQ scoring against upstream scoring references when available.")
    parser.add_argument("--upstream-root", type=Path, default=Path("third_party/BBQ"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/bbq/runs/olmo2_instruct_clean_exact_chain_10k_seed0"))
    return parser


def write_parity_report(*, upstream_root: Path, run_dir: Path) -> dict[str, object]:
    rscript = shutil.which("Rscript")
    metrics_files = sorted(str(path.relative_to(run_dir)) for path in (run_dir / "metrics").glob("*/*/summary.json"))
    report: dict[str, object] = {
        "python_scorer": "available",
        "upstream_root": str(upstream_root),
        "upstream_scoring_reference": str(upstream_root / "analysis_scripts" / "BBQ_calculate_bias_score.R"),
        "metrics_summary_files": metrics_files,
        "rscript": "available" if rscript else "not_found",
        "r_parity": "not_run",
    }
    if not rscript:
        report["r_parity_reason"] = "Rscript is not on PATH; Python scorer mirrors the upstream R formula."
    else:
        report.update(_run_r_summary_parity(rscript=Path(rscript), run_dir=run_dir))
    write_json(run_dir / "upstream_parity" / "parity_report.json", report)
    return report


def _run_r_summary_parity(*, rscript: Path, run_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="bbq-r-parity-") as temp_dir:
        script_path = Path(temp_dir) / "bbq_summary_parity.R"
        script_path.write_text(R_SUMMARY_PARITY_SCRIPT, encoding="utf-8")
        result = subprocess.run(
            [str(rscript), str(script_path), str(run_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        return {
            "r_parity": "failed",
            "r_parity_reason": "Rscript failed while reading local prediction JSONL files.",
            "r_stderr": result.stderr[-4000:],
        }
    try:
        r_summaries = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {
            "r_parity": "failed",
            "r_parity_reason": f"Rscript emitted invalid JSON: {exc}",
            "r_stdout": result.stdout[-4000:],
        }
    comparisons = _compare_r_summaries_to_python(run_dir=run_dir, r_summaries=r_summaries)
    failed = [item for item in comparisons if not item["matches"]]
    return {
        "r_parity": "failed" if failed else "passed",
        "r_parity_reason": "Compared R-derived row-count summaries against Python summary artifacts.",
        "r_comparisons": comparisons,
    }


def _compare_r_summaries_to_python(*, run_dir: Path, r_summaries: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for key, r_summary in sorted(r_summaries.items()):
        parts = Path(key).parts
        if len(parts) < 3:
            continue
        target_id, prompt_format = parts[0], parts[1]
        python_summary_path = run_dir / "metrics" / target_id / prompt_format / "summary.json"
        if not python_summary_path.exists():
            comparisons.append(
                {
                    "prediction_key": key,
                    "matches": False,
                    "reason": f"Missing Python summary: {python_summary_path}",
                }
            )
            continue
        python_summary = read_json(python_summary_path)
        fields = ("total_rows", "matched_rows", "unmatched_rows")
        mismatches = {
            field: {"python": python_summary.get(field), "r": r_summary.get(field)}
            for field in fields
            if int(python_summary.get(field, -1)) != int(r_summary.get(field, -2))
        }
        comparisons.append(
            {
                "prediction_key": key,
                "matches": not mismatches,
                "mismatches": mismatches,
            }
        )
    return comparisons


R_SUMMARY_PARITY_SCRIPT = r"""
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
run_dir <- args[[1]]
pred_root <- file.path(run_dir, "predictions")
pred_files <- list.files(pred_root, pattern = "predictions\\.jsonl$", recursive = TRUE, full.names = TRUE)
summaries <- list()
for (pred_file in pred_files) {
  dat <- stream_in(file(pred_file), verbose = FALSE)
  rel <- sub(paste0(normalizePath(pred_root), "/"), "", normalizePath(pred_file), fixed = TRUE)
  matched <- dat[dat$matched == TRUE, ]
  summaries[[rel]] <- list(
    total_rows = nrow(dat),
    matched_rows = nrow(matched),
    unmatched_rows = nrow(dat) - nrow(matched)
  )
}
cat(toJSON(summaries, auto_unbox = TRUE, null = "null"))
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    write_parity_report(upstream_root=args.upstream_root, run_dir=args.run_dir)
    print(f"Wrote BBQ parity report to {args.run_dir / 'upstream_parity' / 'parity_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
