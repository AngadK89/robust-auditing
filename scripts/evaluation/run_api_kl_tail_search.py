from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.model_equality.api_kl_tail_search import (
    DEFAULT_API_KL_TAIL_VARIANTS,
    ApiKlTailVariant,
    KNOWN_API_KL_TAIL_VARIANTS,
    build_api_kl_training_root,
    summarize_api_kl_tail_search,
    variant_by_name,
)
from robust_auditing.model_equality.section5 import DEFAULT_MODEL_ID


DEFAULT_REFERENCE_ROOT = Path(
    "artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64"
)
DEFAULT_SOURCE_ADAPTER_DIR = Path("outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter")
DEFAULT_OUTPUT_ROOT = Path("artifacts/model_equality_section5/api_kl_tail_search_seed0")
DEFAULT_ADAPTER_OUTPUT_ROOT = Path("outputs/targeted_ft")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate API-faithful KL-tail MET adapters.")
    parser.add_argument("--phase", choices=("prepare", "train", "evaluate", "summarize", "all"), default="all")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--source-adapter-dir", type=Path, default=DEFAULT_SOURCE_ADAPTER_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--adapter-output-root", type=Path, default=DEFAULT_ADAPTER_OUTPUT_ROOT)
    parser.add_argument("--variant", action="append", choices=tuple(variant.name for variant in KNOWN_API_KL_TAIL_VARIANTS))
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--trace-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-type", choices=("kl", "sft", "sft_kl"), default="kl")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--prompt-format", choices=("auto", "raw", "chat"), default="chat")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--eval-device", choices=("cuda", "mps", "cpu", "auto"), default="cuda")
    parser.add_argument("--generation-backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--bank-samples-per-prompt", type=int, default=250)
    parser.add_argument("--sample-multiplier", type=int, default=10)
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--secondary-alpha", type=float, default=0.01)
    parser.add_argument("--failure-rejection-rate", type=float, default=0.5)
    parser.add_argument("--overwrite-train", action="store_true")
    parser.add_argument("--overwrite-eval", action="store_true")
    parser.add_argument(
        "--force-all-variants",
        action="store_true",
        help="Train/evaluate the fallback s150/k20 variant even if an earlier variant passes alpha 0.05.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    variants = variant_by_name(args.variant)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(args, variants)

    if args.phase in {"prepare", "all"}:
        prepare_training_roots(args, variants)
    if args.phase == "prepare":
        return 0

    if args.phase == "all":
        prepare_training_roots(args, variants)
        first_three = variants[:3]
        for variant in first_three:
            train_variant(args, variant)
            evaluate_variant(args, variant)
            summarize(args, variants)
        if len(variants) > 3:
            summary = summarize(args, variants)
            if args.force_all_variants or summary.get("best_variant_alpha_0_05") is None:
                for variant in variants[3:]:
                    train_variant(args, variant)
                    evaluate_variant(args, variant)
                    summarize(args, variants)
        summarize(args, variants)
        return 0

    if args.phase == "train":
        prepare_training_roots(args, variants)
        for variant in variants:
            train_variant(args, variant)
        return 0

    if args.phase == "evaluate":
        for variant in variants:
            evaluate_variant(args, variant)
        return 0

    if args.phase == "summarize":
        summarize(args, variants)
        return 0

    raise ValueError(f"Unhandled phase: {args.phase}")


def prepare_training_roots(args: argparse.Namespace, variants: Sequence[ApiKlTailVariant]) -> None:
    prepared: set[str] = set()
    for variant in variants:
        if variant.train_root_key in prepared:
            continue
        prepared.add(variant.train_root_key)
        train_root = args.output_root / "train_roots" / variant.train_root_key
        manifest = build_api_kl_training_root(
            reference_root=args.reference_root,
            output_root=train_root,
            traces_per_prompt=variant.training_trace_spec,
            trace_seed=args.trace_seed,
            source_adapter_dir=args.source_adapter_dir,
        )
        print(json.dumps({"phase": "prepare", "train_root": str(train_root), **_manifest_counts(manifest)}, sort_keys=True))


def train_variant(args: argparse.Namespace, variant: ApiKlTailVariant) -> None:
    output_dir = adapter_output_dir(args, variant)
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists() and (output_dir / "metrics.json").exists() and not args.overwrite_train:
        print(json.dumps({"phase": "train", "variant": variant.name, "status": "skipped_existing"}), flush=True)
        return
    train_root = args.output_root / "train_roots" / variant.train_root_key
    command = [
        sys.executable,
        "-m",
        "robust_auditing.model_equality.preservation_sft",
        "--source-adapter-dir",
        str(args.source_adapter_dir),
        "--met-root",
        str(train_root),
        "--output-dir",
        str(output_dir),
        "--base-model-id",
        str(args.base_model_id),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--max-length",
        str(args.max_length),
        "--dtype",
        args.dtype,
        "--prompt-format",
        args.prompt_format,
        "--seed",
        str(args.seed),
        "--logging-steps",
        str(args.logging_steps),
        "--loss-type",
        args.loss_type,
        "--max-steps",
        str(variant.max_steps),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
    ]
    _run(command)


def evaluate_variant(args: argparse.Namespace, variant: ApiKlTailVariant) -> None:
    eval_dir = eval_output_dir(args, variant)
    if (eval_dir / "summary.json").exists() and not args.overwrite_eval:
        print(json.dumps({"phase": "evaluate", "variant": variant.name, "status": "skipped_existing"}), flush=True)
        return
    adapter_dir = adapter_output_dir(args, variant) / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter does not exist for {variant.name}: {adapter_dir}")
    command = [
        sys.executable,
        "scripts/evaluation/run_section5_cached_adapter_met.py",
        "--base-model-id",
        str(args.base_model_id),
        "--reference-root",
        str(args.reference_root),
        "--adapter-dir",
        str(adapter_dir),
        "--candidate-label",
        variant.name,
        "--candidate-model-alias",
        "q",
        "--output-root",
        str(eval_dir),
        "--bank-samples-per-prompt",
        str(args.bank_samples_per_prompt),
        "--sample-multiplier",
        str(args.sample_multiplier),
        "--n-simulations",
        str(args.n_simulations),
        "--bootstrap-draws",
        str(args.bootstrap_draws),
        "--alpha",
        str(args.alpha),
        "--secondary-alpha",
        str(args.secondary_alpha),
        "--failure-rejection-rate",
        str(args.failure_rejection_rate),
        "--batch-size",
        str(args.eval_batch_size),
        "--dtype",
        args.eval_dtype,
        "--device",
        args.eval_device,
        "--generation-backend",
        args.generation_backend,
        "--prompt-format",
        "raw",
    ]
    if args.overwrite_eval:
        command.append("--overwrite-q-banks")
    if args.no_progress:
        command.append("--no-progress")
    _run(command)


def summarize(args: argparse.Namespace, variants: Sequence[ApiKlTailVariant]) -> dict[str, object]:
    adapter_dirs = {variant.name: adapter_output_dir(args, variant) / "adapter" for variant in variants}
    summary = summarize_api_kl_tail_search(
        search_root=args.output_root,
        variants=variants,
        adapter_dirs=adapter_dirs,
        failure_rejection_rate=args.failure_rejection_rate,
    )
    print(
        json.dumps(
            {
                "phase": "summarize",
                "best_variant_alpha_0_05": summary.get("best_variant_alpha_0_05"),
                "best_variant_alpha_0_01": summary.get("best_variant_alpha_0_01"),
                "summary_csv": str(args.output_root / "summary.csv"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def adapter_output_dir(args: argparse.Namespace, variant: ApiKlTailVariant) -> Path:
    return args.adapter_output_root / variant.adapter_output_name


def eval_output_dir(args: argparse.Namespace, variant: ApiKlTailVariant) -> Path:
    return args.output_root / "evals" / variant.name


def _write_run_manifest(args: argparse.Namespace, variants: Sequence[ApiKlTailVariant]) -> None:
    payload = {
        "reference_root": str(args.reference_root),
        "source_adapter_dir": str(args.source_adapter_dir),
        "output_root": str(args.output_root),
        "adapter_output_root": str(args.adapter_output_root),
        "base_model_id": str(args.base_model_id),
        "trace_seed": args.trace_seed,
        "seed": args.seed,
        "variants": [variant.to_json() for variant in variants],
        "training": {
            "loss_type": args.loss_type,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "prompt_format": args.prompt_format,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
        },
        "evaluation": {
            "bank_samples_per_prompt": args.bank_samples_per_prompt,
            "sample_multiplier": args.sample_multiplier,
            "n_simulations": args.n_simulations,
            "bootstrap_draws": args.bootstrap_draws,
            "alpha": args.alpha,
            "secondary_alpha": args.secondary_alpha,
            "failure_rejection_rate": args.failure_rejection_rate,
            "batch_size": args.eval_batch_size,
            "dtype": args.eval_dtype,
            "generation_backend": args.generation_backend,
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _manifest_counts(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "traces_per_prompt": manifest["traces_per_prompt"],
        "traces_per_prompt_by_suite": manifest["traces_per_prompt_by_suite"],
        "total_train_record_count": manifest["total_train_record_count"],
        "train_record_count_by_suite": manifest["train_record_count_by_suite"],
    }


def _run(command: Sequence[str]) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(json.dumps({"command": list(command)}, sort_keys=True), flush=True)
    subprocess.run(list(command), cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
