from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence

from robust_auditing.bbq.artifacts import DEFAULT_BBQ_ARTIFACT_ROOT, DEFAULT_RUN_ID, read_jsonl, write_json
from robust_auditing.bbq.formatting import SUPPORTED_FORMATS
from robust_auditing.bbq.targets import DEFAULT_TARGET_IDS, resolve_targets
from scripts.bbq import check_upstream_scoring_parity, generate_predictions, sample_subset, score_results


PredictionRunner = Callable[..., None]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OLMo2 BBQ 10k evaluation.")
    parser.add_argument("--upstream-root", type=Path, default=Path("third_party/BBQ"))
    parser.add_argument("--subset-id", default="10k_seed0")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGET_IDS))
    parser.add_argument("--formats", nargs="+", default=list(SUPPORTED_FORMATS), choices=SUPPORTED_FORMATS)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_BBQ_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser


def run_benchmark(
    *,
    upstream_root: Path,
    subset_id: str,
    targets: Sequence[str],
    formats: Sequence[str],
    max_examples: int = 10_000,
    seed: int = 0,
    artifact_root: Path = DEFAULT_BBQ_ARTIFACT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
    dtype: str = "bf16",
    device_map: str = "auto",
    batch_size: int = 8,
    max_new_tokens: int = 16,
    prediction_runner: PredictionRunner | None = None,
) -> dict[str, Any]:
    resolve_targets(targets)
    subset_dir = artifact_root / "subsets" / subset_id
    subset_path = subset_dir / "examples.jsonl"
    if not subset_path.exists():
        sample_subset.write_subset(
            upstream_root=upstream_root,
            subset_id=subset_id,
            max_examples=max_examples,
            seed=seed,
            output_root=artifact_root / "subsets",
        )
    subset_rows = read_jsonl(subset_path)
    run_dir = artifact_root / "runs" / run_id
    config = {
        "upstream_root": str(upstream_root),
        "subset_id": subset_id,
        "subset_path": str(subset_path),
        "run_dir": str(run_dir),
        "targets": list(targets),
        "formats": list(formats),
        "dtype": dtype,
        "device_map": device_map,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "max_examples": max_examples,
        "seed": seed,
    }
    write_json(run_dir / "config.json", config)
    runner = prediction_runner or generate_predictions.generate_predictions
    runner(
        subset_rows=subset_rows,
        run_dir=run_dir,
        targets=tuple(targets),
        formats=tuple(formats),
        dtype=dtype,
        device_map=device_map,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    generate_predictions.write_compat_exports(run_dir=run_dir, targets=tuple(targets), formats=tuple(formats))
    score_summary = score_results.score_run(run_dir=run_dir, targets=tuple(targets), formats=tuple(formats))
    parity_report = check_upstream_scoring_parity.write_parity_report(upstream_root=upstream_root, run_dir=run_dir)
    summary = {
        **config,
        "example_count": len(subset_rows),
        "score_summary": score_summary,
        "parity_report": parity_report,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_benchmark(
        upstream_root=args.upstream_root,
        subset_id=args.subset_id,
        targets=tuple(args.targets),
        formats=tuple(args.formats),
        max_examples=args.max_examples,
        seed=args.seed,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        dtype=args.dtype,
        device_map=args.device_map,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Wrote BBQ run artifacts to {summary['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
