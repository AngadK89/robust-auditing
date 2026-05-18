#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.evaluation.adapter_suite import (  # noqa: E402
    AdapterSuiteConfig,
    cleanup_memory,
    cleanup_model,
    copy_subset_prompts,
    derive_medmcqa_eval_ids,
    derive_run_id,
    load_adapter_model,
    run_proflingo_verification,
    utc_now,
    write_json,
)
from robust_auditing.fairness.generation import GenerationConfig, generate_responses_for_audit  # noqa: E402
from robust_auditing.fairness.artifacts import read_jsonl, write_jsonl  # noqa: E402
from robust_auditing.fairness.scoring import ScoringConfig, score_audit  # noqa: E402
from robust_auditing.fairness.cli import METRIC_FACTORIES  # noqa: E402
from robust_auditing.medmcqa_rlvr.train import set_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen an adapter with ProFLingo and held-out BOLD only.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/adapter_evals"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--skip-proflingo", action="store_true")
    parser.add_argument("--proflingo-limit", type=int, default=None)
    parser.add_argument(
        "--bold-smoke-examples",
        type=int,
        default=None,
        help="Evaluate a stratified sample from the stored 10k BOLD subset for faster smoke iteration.",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    config = AdapterSuiteConfig(
        adapter_dir=args.adapter_dir,
        run_id=derive_run_id(args.adapter_dir),
        output_root=args.output_root,
        medmcqa_eval_ids=derive_medmcqa_eval_ids(args.adapter_dir),
        fairness_subset_id=_bold_subset_id(args.bold_smoke_examples, seed=0),
        dtype=args.dtype,
        device_map=args.device_map,
        fairness_batch_size=args.batch_size,
        classifier_batch_size=args.classifier_batch_size,
        proflingo_limit=args.proflingo_limit,
        skip_generation_eval=True,
    )
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "bold_only_config.json", config.to_json())
    started_at = utc_now()
    model, tokenizer, proflingo_tokenizer = load_adapter_model(config)
    summary: dict[str, object] = {
        "started_at": started_at,
        "adapter": {
            "adapter_dir": str(config.adapter_dir),
            "run_id": config.run_id,
            "base_model_id": config.base_model_id,
        },
        "config": config.to_json(),
    }
    try:
        if not args.skip_proflingo:
            summary["proflingo"] = run_proflingo_verification(config, model, proflingo_tokenizer)
        if args.bold_smoke_examples is None:
            copy_subset_prompts(config)
        else:
            _write_bold_smoke_subset(config, max_examples=args.bold_smoke_examples)
        generation_config = GenerationConfig(
            audits=("bold",),
            model_id=config.fairness_model_id,
            batch_size=config.fairness_batch_size,
            output_root=config.fairness_output_root,
            dtype=config.dtype,
            device_map=config.device_map,
            seed=config.seed,
            subset_id=config.fairness_subset_id,
        )
        bold_dir = generate_responses_for_audit("bold", generation_config, model=model, tokenizer=tokenizer)
        summary["fairness_generation"] = {"bold": str(bold_dir)}
    finally:
        cleanup_model(model)
        cleanup_memory()

    scoring_config = ScoringConfig(
        audits=("bold",),
        model_id=config.fairness_model_id,
        batch_size=config.classifier_batch_size,
        group_by=("axis", "bucket"),
        output_root=config.fairness_output_root,
        metric="bold_stddev_toxicity_metric",
        subset_id=config.fairness_subset_id,
    )
    metric = METRIC_FACTORIES["bold_stddev_toxicity_metric"].from_config(scoring_config)
    metric_dir = score_audit("bold", scoring_config, metric=metric)
    metadata = json.loads((metric_dir / "metadata.json").read_text())
    summary["fairness"] = {
        "bold": {
            "metric": "bold_stddev_toxicity_metric",
            "bold_stddev_toxicity_metric": metadata["bold_stddev_toxicity_metric"],
            "overall_mean_sentiment": metadata["overall_mean_sentiment"],
            "overall_mean_toxicity": metadata["overall_mean_toxicity"],
            "metadata_path": str(metric_dir / "metadata.json"),
            "metric_dir": str(metric_dir),
        }
    }
    summary["finished_at"] = utc_now()
    write_json(config.output_dir / "bold_only_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _bold_subset_id(max_examples: int | None, *, seed: int) -> str:
    if max_examples is None:
        return "10k_seed0"
    return f"10k_seed0_bold{max_examples}_seed{seed}"


def _write_bold_smoke_subset(config: AdapterSuiteConfig, *, max_examples: int) -> None:
    if max_examples <= 0:
        raise ValueError("--bold-smoke-examples must be positive")
    source = Path("artifacts/fairness") / "bold" / "10k_seed0" / "normalized_prompts.jsonl"
    destination = config.fairness_output_root / "bold" / config.fairness_subset_id / "normalized_prompts.jsonl"
    if destination.exists():
        return
    if not source.exists():
        raise FileNotFoundError(f"Missing source BOLD subset prompts: {source}")
    rows = _stratified_sample_bold_rows(read_jsonl(source), max_examples=max_examples, seed=config.seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination, rows)
    logging.info("Wrote %d stratified BOLD smoke prompts to %s", len(rows), destination)


def _stratified_sample_bold_rows(
    rows: list[dict[str, object]],
    *,
    max_examples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("axis", "")), str(row.get("bucket", "")))].append(row)
    for group_rows in grouped.values():
        rng.shuffle(group_rows)
    keys = sorted(grouped)
    if max_examples >= len(rows):
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled

    allocations = _proportional_group_allocations({key: len(grouped[key]) for key in keys}, max_examples)
    sampled: list[dict[str, object]] = []
    for key in keys:
        count = allocations.get(key, 0)
        if count > 0:
            sampled.extend(grouped[key][:count])
    rng.shuffle(sampled)
    return sampled


def _proportional_group_allocations(group_sizes: dict[tuple[str, str], int], max_examples: int) -> dict[tuple[str, str], int]:
    """Allocate a smoke subset proportionally while retaining broad descriptor coverage."""

    if max_examples <= 0:
        return {}
    positive = {key: size for key, size in group_sizes.items() if size > 0}
    if not positive:
        return {}
    if max_examples < len(positive):
        return {key: 1 for key, _ in sorted(positive.items(), key=lambda item: (-item[1], item[0]))[:max_examples]}

    total = sum(positive.values())
    allocations: dict[tuple[str, str], int] = {key: 1 for key in positive}
    remaining = max_examples - len(positive)
    weights = {key: max(size - 1, 0) for key, size in positive.items()}
    weight_total = sum(weights.values())
    if remaining <= 0 or weight_total == 0:
        return allocations

    remainders: list[tuple[float, tuple[str, str]]] = []
    allocated_extra = 0
    for key, weight in weights.items():
        raw = remaining * (weight / weight_total)
        extra = int(raw)
        allocations[key] += extra
        allocated_extra += extra
        remainders.append((raw - extra, key))
    for _, key in sorted(remainders, reverse=True)[: remaining - allocated_extra]:
        allocations[key] += 1

    for key, size in positive.items():
        allocations[key] = min(allocations[key], size)
    shortfall = max_examples - sum(allocations.values())
    if shortfall > 0:
        for _, key in sorted(((positive[key] - allocations[key], key) for key in positive), reverse=True):
            if shortfall <= 0:
                break
            room = positive[key] - allocations[key]
            add = min(room, shortfall)
            allocations[key] += add
            shortfall -= add
    return allocations


if __name__ == "__main__":
    raise SystemExit(main())
