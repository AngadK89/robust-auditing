from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from robust_auditing.bbq.artifacts import read_jsonl, write_jsonl
from robust_auditing.bbq.formatting import SUPPORTED_FORMATS
from robust_auditing.bbq.inference import BBQGenerationConfig, generate_predictions_for_target_formats
from robust_auditing.bbq.targets import DEFAULT_TARGET_IDS, resolve_targets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate OLMo2 BBQ predictions for a prepared subset.")
    parser.add_argument("--subset", type=Path, default=Path("artifacts/bbq/subsets/10k_seed0/examples.jsonl"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/bbq/runs/olmo2_instruct_clean_exact_chain_10k_seed0"))
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGET_IDS))
    parser.add_argument("--formats", nargs="+", default=list(SUPPORTED_FORMATS), choices=SUPPORTED_FORMATS)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def generate_predictions(
    *,
    subset_rows: Sequence[dict[str, Any]],
    run_dir: Path,
    targets: Sequence[str],
    formats: Sequence[str],
    dtype: str = "bf16",
    device_map: str = "auto",
    batch_size: int = 8,
    max_new_tokens: int = 16,
    seed: int = 0,
) -> None:
    config = BBQGenerationConfig(
        dtype=dtype,
        device_map=device_map,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    for target in resolve_targets(targets):
        records_by_format = generate_predictions_for_target_formats(
            target,
            subset_rows,
            prompt_formats=tuple(formats),
            config=config,
        )
        for prompt_format, records in records_by_format.items():
            write_jsonl(run_dir / "predictions" / target.target_id / prompt_format / "predictions.jsonl", records)
    write_compat_exports(run_dir=run_dir, targets=targets, formats=formats)


def write_compat_exports(*, run_dir: Path, targets: Sequence[str], formats: Sequence[str]) -> None:
    for target_id in targets:
        by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
        for prompt_format in formats:
            path = run_dir / "predictions" / target_id / prompt_format / "predictions.jsonl"
            if not path.exists():
                continue
            for row in read_jsonl(path):
                key = (str(row["category"]), int(row["example_id"]), str(row["question_index"]))
                compat = by_key.setdefault(key, _base_compat_row(row))
                compat[f"{target_id}_pred_{prompt_format}"] = row["raw_output"]
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_key.values():
            by_category[str(row["category"])].append(row)
        for category, rows in by_category.items():
            rows = sorted(rows, key=lambda row: (int(row["example_id"]), str(row["question_index"])))
            write_jsonl(run_dir / "predictions_compat" / target_id / f"{category}.jsonl", rows)


def _base_compat_row(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "target_id",
        "format",
        "prompt",
        "raw_output",
        "normalized_output",
        "pred_label",
        "pred_cat",
        "matched",
        "is_unknown",
        "is_correct",
        "is_biased_answer",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = read_jsonl(args.subset)
    generate_predictions(
        subset_rows=rows,
        run_dir=args.run_dir,
        targets=tuple(args.targets),
        formats=tuple(args.formats),
        dtype=args.dtype,
        device_map=args.device_map,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(f"Wrote BBQ predictions to {args.run_dir / 'predictions'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
