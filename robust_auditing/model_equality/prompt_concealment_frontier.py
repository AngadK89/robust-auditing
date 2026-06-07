from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from robust_auditing.model_equality.api_kl_tail_search import build_api_kl_training_root
from robust_auditing.model_equality import section5
from robust_auditing.model_equality.completions import CompletionRecord, read_completion_records, write_completion_records
from robust_auditing.model_equality.constants import DEFAULT_MODEL_ID
from robust_auditing.model_equality.generation import CompletionGenerator, GenerationRuntimeConfig
from robust_auditing.model_equality.prompts import PromptRecord, write_prompt_records


DEFAULT_REFERENCE_ROOT = Path(
    "artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_20260606_b64"
)
DEFAULT_SELECTED_TRAIN_ROOT = Path(
    "artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/train_roots/w020_h020_u040"
)
DEFAULT_SOURCE_ADAPTER_DIR = Path("outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter")
DEFAULT_OUTPUT_ROOT = Path("artifacts/model_equality_section5/prompt_concealment_frontier_u40_s150_seed0_9")
DEFAULT_BASELINE_FRONTIER_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_ADAPTER_OUTPUT_ROOT = Path("outputs/targeted_ft")
DEFAULT_ENDPOINT_U40_SUMMARY = Path(
    "artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/evals/"
    "api_met_kl_s150_w20_h20_u40/summary.json"
)
DEFAULT_ENDPOINT_POISONED_SUMMARY = Path(
    "artifacts/model_equality_section5/faithful_instruct_clean_poisoned_20260606_b64/summary.json"
)
DEFAULT_PROMPT_SUITES = ("wikipedia_en", "humaneval", "ultrachat")
DEFAULT_HIDDEN_LEVELS = (25, 50, 75)
DEFAULT_SPLIT_SEEDS = tuple(range(10))
DEFAULT_TRACE_COUNTS_BY_SUITE = {"wikipedia_en": 20, "humaneval": 20, "ultrachat": 40}
DEFAULT_ADAPTER_RUN_PREFIX = "prompt_conceal_u40_s150"
DEFAULT_FIGURE_PREFIX = "met_prompt_concealment"
REFERENCE_BANK_FILENAMES = ("completion_bank_p.jsonl", "completion_bank_olmo-instruct.jsonl")


@dataclass(frozen=True)
class PromptConcealmentSplit:
    split_seed: int
    hidden_level: int
    prompt_ids_by_suite: dict[str, list[str]]
    prompt_order_by_suite: dict[str, list[str]]
    hidden_prompt_ids_by_suite: dict[str, list[str]]
    visible_prompt_ids_by_suite: dict[str, list[str]]

    @property
    def hidden_prompt_count(self) -> int:
        return sum(len(ids) for ids in self.hidden_prompt_ids_by_suite.values())

    @property
    def visible_prompt_count(self) -> int:
        return sum(len(ids) for ids in self.visible_prompt_ids_by_suite.values())

    @property
    def total_prompt_count(self) -> int:
        return sum(len(ids) for ids in self.prompt_ids_by_suite.values())


@dataclass(frozen=True)
class PromptConcealmentConfig:
    reference_root: Path = DEFAULT_REFERENCE_ROOT
    selected_train_root: Path = DEFAULT_SELECTED_TRAIN_ROOT
    source_adapter_dir: Path = DEFAULT_SOURCE_ADAPTER_DIR
    output_root: Path = DEFAULT_OUTPUT_ROOT
    adapter_output_root: Path = DEFAULT_ADAPTER_OUTPUT_ROOT
    adapter_run_prefix: str = DEFAULT_ADAPTER_RUN_PREFIX
    figure_prefix: str = DEFAULT_FIGURE_PREFIX
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS
    hidden_levels: tuple[int, ...] = DEFAULT_HIDDEN_LEVELS
    prompt_suites: tuple[str, ...] = DEFAULT_PROMPT_SUITES
    selected_trace_counts_by_suite: dict[str, int] | None = None
    base_model_id: str = DEFAULT_MODEL_ID
    learning_rate: float = 1e-5
    max_steps: int = 150
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 1536
    dtype: str = "bf16"
    prompt_format: str = "chat"
    seed: int = 0
    trace_seed: int = 0
    loss_type: str = "kl"
    kl_temperature: float = 1.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 25
    bank_samples_per_prompt: int = 250
    sample_multiplier: int = 10
    n_simulations: int = 100
    bootstrap_draws: int = 1000
    alpha: float = 0.05
    secondary_alpha: float = 0.05
    failure_rejection_rate: float = 0.5
    effect_repeats: int = 10
    effect_sample_multiplier: int = 100
    q_generation_batch_size: int = 64
    eval_dtype: str = "bf16"
    eval_device: str = "cuda"
    generation_backend: str = "hf"
    max_num_seqs: int = 1024
    gpu_memory_utilization: float = 0.95
    overwrite_train: bool = False
    overwrite_eval: bool = False
    overwrite_q_banks: bool = False
    progress: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PromptConcealmentConfig":
        split_seeds = tuple(args.split_seed) if args.split_seed else DEFAULT_SPLIT_SEEDS
        hidden_levels = tuple(args.hidden_level) if args.hidden_level else DEFAULT_HIDDEN_LEVELS
        prompt_suites = tuple(args.prompt_suite) if args.prompt_suite else DEFAULT_PROMPT_SUITES
        return cls(
            reference_root=Path(args.reference_root),
            selected_train_root=Path(args.selected_train_root),
            source_adapter_dir=Path(args.source_adapter_dir),
            output_root=Path(args.output_root),
            adapter_output_root=Path(args.adapter_output_root),
            adapter_run_prefix=args.adapter_run_prefix,
            figure_prefix=args.figure_prefix,
            split_seeds=split_seeds,
            hidden_levels=hidden_levels,
            prompt_suites=prompt_suites,
            selected_trace_counts_by_suite=_parse_selected_trace_counts(args.selected_trace_count, prompt_suites),
            base_model_id=args.base_model_id,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            dtype=args.dtype,
            prompt_format=args.prompt_format,
            seed=args.seed,
            trace_seed=args.trace_seed,
            loss_type=args.loss_type,
            kl_temperature=args.kl_temperature,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            bank_samples_per_prompt=args.bank_samples_per_prompt,
            sample_multiplier=args.sample_multiplier,
            n_simulations=args.n_simulations,
            bootstrap_draws=args.bootstrap_draws,
            alpha=args.alpha,
            secondary_alpha=args.secondary_alpha,
            failure_rejection_rate=args.failure_rejection_rate,
            effect_repeats=args.effect_repeats,
            effect_sample_multiplier=args.effect_sample_multiplier,
            q_generation_batch_size=args.q_generation_batch_size,
            eval_dtype=args.eval_dtype,
            eval_device=args.eval_device,
            generation_backend=args.generation_backend,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            overwrite_train=args.overwrite_train,
            overwrite_eval=args.overwrite_eval,
            overwrite_q_banks=args.overwrite_q_banks,
            progress=not args.no_progress,
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "reference_root",
            "selected_train_root",
            "source_adapter_dir",
            "output_root",
            "adapter_output_root",
        ):
            payload[key] = str(payload[key])
        if payload["selected_trace_counts_by_suite"] is not None:
            payload["selected_trace_counts_by_suite"] = dict(sorted(payload["selected_trace_counts_by_suite"].items()))
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Section 6.3 MET prompt-concealment frontier experiment.")
    parser.add_argument("--phase", choices=("prepare", "train", "evaluate", "summarize", "all"), default="all")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--selected-train-root", type=Path, default=DEFAULT_SELECTED_TRAIN_ROOT)
    parser.add_argument("--source-adapter-dir", type=Path, default=DEFAULT_SOURCE_ADAPTER_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--adapter-output-root", type=Path, default=DEFAULT_ADAPTER_OUTPUT_ROOT)
    parser.add_argument("--adapter-run-prefix", default=DEFAULT_ADAPTER_RUN_PREFIX)
    parser.add_argument("--figure-prefix", default=DEFAULT_FIGURE_PREFIX)
    parser.add_argument("--split-seed", action="append", type=int)
    parser.add_argument("--hidden-level", action="append", type=int)
    parser.add_argument("--prompt-suite", action="append", choices=tuple(section5.SECTION5_SUITE_SPECS))
    parser.add_argument(
        "--selected-trace-count",
        action="append",
        default=[],
        metavar="SUITE=COUNT",
        help="Build/validate selected train root with suite-specific traces per prompt, e.g. wikipedia_en=80.",
    )
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--prompt-format", choices=("auto", "raw", "chat"), default="chat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-seed", type=int, default=0)
    parser.add_argument("--loss-type", choices=("kl", "sft", "sft_kl"), default="kl")
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--bank-samples-per-prompt", type=int, default=250)
    parser.add_argument("--sample-multiplier", type=int, default=10)
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--secondary-alpha", type=float, default=0.05)
    parser.add_argument("--failure-rejection-rate", type=float, default=0.5)
    parser.add_argument("--effect-repeats", type=int, default=10)
    parser.add_argument("--effect-sample-multiplier", type=int, default=100)
    parser.add_argument("--q-generation-batch-size", type=int, default=64)
    parser.add_argument("--eval-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--eval-device", choices=("cuda", "mps", "cpu", "auto"), default="cuda")
    parser.add_argument("--generation-backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--endpoint-u40-summary", type=Path, default=DEFAULT_ENDPOINT_U40_SUMMARY)
    parser.add_argument("--endpoint-poisoned-summary", type=Path, default=DEFAULT_ENDPOINT_POISONED_SUMMARY)
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--overwrite-train", action="store_true")
    parser.add_argument("--overwrite-eval", action="store_true")
    parser.add_argument("--overwrite-q-banks", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def build_prompt_concealment_splits(
    reference_root: Path,
    *,
    split_seed: int,
    hidden_levels: Sequence[int] = (*DEFAULT_HIDDEN_LEVELS, 0, 100),
    prompt_suites: Sequence[str] = DEFAULT_PROMPT_SUITES,
) -> dict[int, PromptConcealmentSplit]:
    prompts_by_suite = load_prompt_records_by_suite(reference_root, prompt_suites=prompt_suites)
    prompt_ids_by_suite = {suite: [record.prompt_id for record in records] for suite, records in prompts_by_suite.items()}
    prompt_order_by_suite: dict[str, list[str]] = {}
    for suite, prompt_ids in prompt_ids_by_suite.items():
        order = list(prompt_ids)
        random.Random(_sha256_seed(f"{int(split_seed)}\0{suite}")).shuffle(order)
        prompt_order_by_suite[suite] = order

    splits: dict[int, PromptConcealmentSplit] = {}
    for hidden_level in hidden_levels:
        hidden_by_suite: dict[str, list[str]] = {}
        visible_by_suite: dict[str, list[str]] = {}
        for suite, order in prompt_order_by_suite.items():
            hidden_count = _hidden_prompt_count(len(order), int(hidden_level))
            hidden_ids = order[:hidden_count]
            visible_ids = order[hidden_count:]
            hidden_by_suite[suite] = list(hidden_ids)
            visible_by_suite[suite] = list(visible_ids)
        splits[int(hidden_level)] = PromptConcealmentSplit(
            split_seed=int(split_seed),
            hidden_level=int(hidden_level),
            prompt_ids_by_suite={suite: list(ids) for suite, ids in prompt_ids_by_suite.items()},
            prompt_order_by_suite={suite: list(ids) for suite, ids in prompt_order_by_suite.items()},
            hidden_prompt_ids_by_suite=hidden_by_suite,
            visible_prompt_ids_by_suite=visible_by_suite,
        )
    return splits


def prepare_concealment_split_artifacts(
    split: PromptConcealmentSplit,
    *,
    reference_root: Path,
    selected_train_root: Path,
    split_root: Path,
    source_adapter_dir: Path,
) -> dict[str, Any]:
    split_root.mkdir(parents=True, exist_ok=True)
    selected_manifest = _read_json(selected_train_root / "split_manifest.json")
    traces_per_prompt_by_suite = {
        suite: int(selected_manifest["traces_per_prompt_by_suite"][suite])
        for suite in split.prompt_ids_by_suite
    }
    train_root = split_root / "train_met_root"
    visible_records_all: list[PromptRecord] = []
    hidden_records_all: list[PromptRecord] = []
    selected_indices_by_suite: dict[str, dict[str, list[int]]] = {}
    train_record_count_by_suite: dict[str, int] = {}
    train_prompt_count_by_suite: dict[str, int] = {}
    hidden_prompt_count_by_suite: dict[str, int] = {}

    for suite in split.prompt_ids_by_suite:
        prompt_records = _read_prompt_records(selected_train_root / "suites" / suite / "prompts.jsonl")
        prompt_rank = {record.prompt_id: index for index, record in enumerate(prompt_records)}
        visible_ids = set(split.visible_prompt_ids_by_suite[suite])
        hidden_ids = set(split.hidden_prompt_ids_by_suite[suite])
        visible_prompts = [record for record in prompt_records if record.prompt_id in visible_ids]
        hidden_prompts = [record for record in prompt_records if record.prompt_id in hidden_ids]
        visible_records_all.extend(visible_prompts)
        hidden_records_all.extend(hidden_prompts)

        suite_train_dir = train_root / "suites" / suite
        _filter_jsonl_by_prompt_ids(
            selected_train_root / "suites" / suite / "prompts.jsonl",
            suite_train_dir / "prompts.jsonl",
            visible_ids,
        )
        _filter_jsonl_by_prompt_ids(
            selected_train_root / "suites" / suite / "completion_bank_p.jsonl",
            suite_train_dir / "completion_bank_p.jsonl",
            visible_ids,
        )

        selected_indices_by_suite[suite] = {}
        records = read_completion_records(suite_train_dir / "completion_bank_p.jsonl")
        for record in records:
            selected_indices_by_suite[suite].setdefault(record.prompt_id, []).append(int(record.sample_index))
        selected_indices_by_suite[suite] = {
            prompt_id: sorted(indices)
            for prompt_id, indices in sorted(
                selected_indices_by_suite[suite].items(),
                key=lambda item: prompt_rank[item[0]],
            )
        }
        train_record_count_by_suite[suite] = len(records)
        train_prompt_count_by_suite[suite] = len(visible_prompts)
        hidden_prompt_count_by_suite[suite] = len(hidden_prompts)

        expected_count = len(visible_prompts) * traces_per_prompt_by_suite[suite]
        if len(records) != expected_count:
            raise ValueError(f"{suite} train record count is {len(records)}, expected {expected_count}")

    _filter_completion_jsonl_by_suite_prompt_ids(
        selected_train_root / "selected_training_completions.jsonl",
        split_root / "selected_training_completions.jsonl",
        {suite: set(ids) for suite, ids in split.visible_prompt_ids_by_suite.items()},
    )
    write_prompt_records(split_root / "visible_prompts.jsonl", visible_records_all)
    write_prompt_records(split_root / "hidden_prompts.jsonl", hidden_records_all)

    total_prompts = split.total_prompt_count
    manifest = {
        "split_seed": split.split_seed,
        "hidden_level": split.hidden_level,
        "hidden_percent_requested": split.hidden_level,
        "hidden_fraction_actual": (split.hidden_prompt_count / total_prompts) if total_prompts else 0.0,
        "hidden_prompt_count": split.hidden_prompt_count,
        "visible_prompt_count": split.visible_prompt_count,
        "train_prompt_count": split.visible_prompt_count,
        "train_record_count": sum(train_record_count_by_suite.values()),
        "reference_root": str(reference_root),
        "selected_train_root": str(selected_train_root),
        "source_adapter_dir": str(source_adapter_dir),
        "split_root": str(split_root),
        "train_met_root": str(train_root),
        "trace_seed": int(selected_manifest.get("trace_seed", 0)),
        "traces_per_prompt_by_suite": traces_per_prompt_by_suite,
        "prompt_ids_by_suite": {suite: list(ids) for suite, ids in split.prompt_ids_by_suite.items()},
        "prompt_order_by_suite": {suite: list(ids) for suite, ids in split.prompt_order_by_suite.items()},
        "hidden_prompt_ids_by_suite": _sort_prompt_ids_by_reference_order(
            split.hidden_prompt_ids_by_suite,
            split.prompt_ids_by_suite,
        ),
        "visible_prompt_ids_by_suite": _sort_prompt_ids_by_reference_order(
            split.visible_prompt_ids_by_suite,
            split.prompt_ids_by_suite,
        ),
        "hidden_prompt_ids_by_suite_split_order": {
            suite: list(ids) for suite, ids in split.hidden_prompt_ids_by_suite.items()
        },
        "visible_prompt_ids_by_suite_split_order": {
            suite: list(ids) for suite, ids in split.visible_prompt_ids_by_suite.items()
        },
        "hidden_prompt_count_by_suite": hidden_prompt_count_by_suite,
        "train_prompt_count_by_suite": train_prompt_count_by_suite,
        "train_record_count_by_suite": train_record_count_by_suite,
        "selected_completion_sample_indices_by_prompt": selected_indices_by_suite,
    }
    _write_json(split_root / "split_manifest.json", manifest)
    return manifest


def run_prepare_phase(config: PromptConcealmentConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    ensure_selected_train_root(config)
    _write_json(config.output_root / "config.json", config.to_json())
    manifests: dict[str, Any] = {}
    for split_seed in config.split_seeds:
        splits = build_prompt_concealment_splits(
            config.reference_root,
            split_seed=split_seed,
            hidden_levels=config.hidden_levels,
            prompt_suites=config.prompt_suites,
        )
        for hidden_level in config.hidden_levels:
            manifest = prepare_concealment_split_artifacts(
                splits[hidden_level],
                reference_root=config.reference_root,
                selected_train_root=config.selected_train_root,
                split_root=split_root_for(config, split_seed=split_seed, hidden_level=hidden_level),
                source_adapter_dir=config.source_adapter_dir,
            )
            manifests[f"split{split_seed:03d}/hidden{hidden_level:03d}"] = manifest
    payload = {"config": config.to_json(), "splits": manifests}
    _write_json(config.output_root / "prepare_summary.json", payload)
    return payload


def run_train_phase(config: PromptConcealmentConfig) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for split_seed in config.split_seeds:
        for hidden_level in config.hidden_levels:
            run_dir = adapter_run_dir(config, split_seed=split_seed, hidden_level=hidden_level)
            adapter_dir = run_dir / "adapter"
            metrics_path = run_dir / "metrics.json"
            key = f"split{split_seed:03d}/hidden{hidden_level:03d}"
            if adapter_dir.exists() and metrics_path.exists() and not config.overwrite_train:
                results[key] = _read_json(metrics_path)
                continue
            manifest_path = split_root_for(config, split_seed=split_seed, hidden_level=hidden_level) / "split_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing split manifest before training: {manifest_path}")
            command = [
                sys.executable,
                "-m",
                "robust_auditing.model_equality.preservation_sft",
                "--source-adapter-dir",
                str(config.source_adapter_dir),
                "--met-root",
                str(split_root_for(config, split_seed=split_seed, hidden_level=hidden_level) / "train_met_root"),
                "--output-dir",
                str(run_dir),
                "--base-model-id",
                str(config.base_model_id),
                "--learning-rate",
                str(config.learning_rate),
                "--batch-size",
                str(config.batch_size),
                "--gradient-accumulation-steps",
                str(config.gradient_accumulation_steps),
                "--max-length",
                str(config.max_length),
                "--dtype",
                config.dtype,
                "--prompt-format",
                config.prompt_format,
                "--seed",
                str(config.seed),
                "--logging-steps",
                str(config.logging_steps),
                "--loss-type",
                config.loss_type,
                "--max-steps",
                str(config.max_steps),
                "--warmup-ratio",
                str(config.warmup_ratio),
                "--weight-decay",
                str(config.weight_decay),
                "--kl-temperature",
                str(config.kl_temperature),
            ]
            print(json.dumps({"phase": "train", "key": key, "command": command}, sort_keys=True), flush=True)
            subprocess.run(command, check=True)
            _copy_json(manifest_path, run_dir / "split_manifest.json")
            results[key] = _read_json(metrics_path)
    _write_json(config.output_root / "train_summary.json", results)
    return results


def run_evaluate_phase(config: PromptConcealmentConfig) -> dict[str, Any]:
    prompts_by_suite = load_prompt_records_by_suite(config.reference_root, prompt_suites=config.prompt_suites)
    p_records_by_suite = load_p_completion_records_by_suite(config.reference_root, prompt_suites=config.prompt_suites)
    results: dict[str, Any] = {}
    for split_seed in config.split_seeds:
        for hidden_level in config.hidden_levels:
            key = f"split{split_seed:03d}/hidden{hidden_level:03d}"
            manifest = _read_json(split_root_for(config, split_seed=split_seed, hidden_level=hidden_level) / "split_manifest.json")
            hidden_ids_by_suite = {
                suite: list(manifest["hidden_prompt_ids_by_suite"].get(suite, []))
                for suite in config.prompt_suites
            }
            adapter_dir = adapter_dir_for(config, split_seed=split_seed, hidden_level=hidden_level)
            if not adapter_dir.exists():
                raise FileNotFoundError(f"Missing trained adapter before evaluation: {adapter_dir}")
            hidden_prompts_by_suite = filter_prompt_records_by_suite(prompts_by_suite, hidden_ids_by_suite)
            q_records_by_suite = ensure_hidden_q_completion_bank(
                config,
                split_seed=split_seed,
                hidden_level=hidden_level,
                adapter_dir=adapter_dir,
                hidden_prompts_by_suite=hidden_prompts_by_suite,
            )
            eval_root = hidden_eval_root_for(config, split_seed=split_seed, hidden_level=hidden_level)
            if (eval_root / "summary.json").exists() and not config.overwrite_eval:
                summary = _read_json(eval_root / "summary.json")
            else:
                summary = section5.run_section5_cached_bank_pipeline(
                    section5_config_for_eval(config, eval_root, adapter_dir, split_seed, hidden_level),
                    prompts_by_suite=prompts_by_suite,
                    p_records_by_suite=p_records_by_suite,
                    q_records_by_suite=q_records_by_suite,
                    prompt_ids_by_suite=hidden_ids_by_suite,
                )
            results[key] = summary
    _write_json(config.output_root / "eval_summary.json", results)
    return results


def summarize_prompt_concealment_results(
    config: PromptConcealmentConfig,
    *,
    endpoint_u40_summary: Path = DEFAULT_ENDPOINT_U40_SUMMARY,
    endpoint_poisoned_summary: Path = DEFAULT_ENDPOINT_POISONED_SUMMARY,
    image_dir: Path | None = Path("images"),
) -> dict[str, Any]:
    per_seed_rows = _load_per_seed_summary_rows(config)
    endpoint_rows, endpoint_decisions = _load_endpoint_rows(
        endpoint_u40_summary=endpoint_u40_summary,
        endpoint_poisoned_summary=endpoint_poisoned_summary,
        prompt_suites=config.prompt_suites,
        failure_rejection_rate=config.failure_rejection_rate,
    )
    all_long_rows = [*endpoint_rows, *per_seed_rows]
    aggregate_rows = _aggregate_suite_rows(all_long_rows)
    decision_rows = _aggregate_decision_rows(per_seed_rows, endpoint_decisions)
    thresholds = _compute_thresholds(aggregate_rows, prompt_suites=config.prompt_suites)

    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv_with_fields(config.output_root / "summary_long.csv", all_long_rows, LONG_FIELDNAMES)
    _write_csv_with_fields(config.output_root / "summary.csv", aggregate_rows, AGGREGATE_FIELDNAMES)
    _write_csv_with_fields(config.output_root / "decision_summary.csv", decision_rows, DECISION_FIELDNAMES)
    figure_path = ""
    decision_figure_path = ""
    if image_dir is not None:
        figure_path = _write_rejection_plot(aggregate_rows, image_dir, figure_prefix=config.figure_prefix)
        decision_figure_path = _write_decision_plot(decision_rows, image_dir, figure_prefix=config.figure_prefix)
    payload = {
        "config": config.to_json(),
        "endpoint_u40_summary": str(endpoint_u40_summary),
        "endpoint_poisoned_summary": str(endpoint_poisoned_summary),
        "per_seed_rows": per_seed_rows,
        "endpoint_rows": endpoint_rows,
        "aggregate_rows": aggregate_rows,
        "decision_rows": decision_rows,
        "thresholds": thresholds,
        "figure_path": figure_path,
        "decision_figure_path": decision_figure_path,
    }
    _write_json(config.output_root / "summary.json", payload)
    comparison_payload = write_trace_ablation_comparison(config, image_dir=image_dir)
    payload["comparison"] = comparison_payload
    _write_json(config.output_root / "summary.json", payload)
    _write_readme(config.output_root, config, figure_path, decision_figure_path)
    return payload


def run_summarize_phase(
    config: PromptConcealmentConfig,
    *,
    endpoint_u40_summary: Path = DEFAULT_ENDPOINT_U40_SUMMARY,
    endpoint_poisoned_summary: Path = DEFAULT_ENDPOINT_POISONED_SUMMARY,
    image_dir: Path | None = Path("images"),
) -> dict[str, Any]:
    return summarize_prompt_concealment_results(
        config,
        endpoint_u40_summary=endpoint_u40_summary,
        endpoint_poisoned_summary=endpoint_poisoned_summary,
        image_dir=image_dir,
    )


def run_pipeline(
    config: PromptConcealmentConfig,
    *,
    phase: str,
    endpoint_u40_summary: Path = DEFAULT_ENDPOINT_U40_SUMMARY,
    endpoint_poisoned_summary: Path = DEFAULT_ENDPOINT_POISONED_SUMMARY,
    image_dir: Path | None = Path("images"),
) -> dict[str, Any]:
    result: dict[str, Any] = {"phase": phase, "output_root": str(config.output_root)}
    if phase in {"prepare", "all"}:
        result = run_prepare_phase(config)
    if phase in {"train", "all"}:
        result = run_train_phase(config)
    if phase in {"evaluate", "all"}:
        result = run_evaluate_phase(config)
    if phase in {"summarize", "all"}:
        result = run_summarize_phase(
            config,
            endpoint_u40_summary=endpoint_u40_summary,
            endpoint_poisoned_summary=endpoint_poisoned_summary,
            image_dir=image_dir,
        )
    return result


def ensure_selected_train_root(config: PromptConcealmentConfig) -> dict[str, Any]:
    manifest_path = config.selected_train_root / "split_manifest.json"
    if config.selected_trace_counts_by_suite is None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing selected train root manifest: {manifest_path}")
        return _read_json(manifest_path)

    expected_counts = {
        suite: int(config.selected_trace_counts_by_suite[suite])
        for suite in config.prompt_suites
    }
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        actual_counts = {
            suite: int(manifest["traces_per_prompt_by_suite"][suite])
            for suite in config.prompt_suites
        }
        if actual_counts != expected_counts:
            raise ValueError(
                f"Selected train root {config.selected_train_root} has trace counts {actual_counts}; "
                f"expected {expected_counts}"
            )
        return manifest

    return build_api_kl_training_root(
        reference_root=config.reference_root,
        output_root=config.selected_train_root,
        traces_per_prompt=expected_counts,
        trace_seed=config.trace_seed,
        prompt_suites=config.prompt_suites,
        source_adapter_dir=config.source_adapter_dir,
    )


def ensure_hidden_q_completion_bank(
    config: PromptConcealmentConfig,
    *,
    split_seed: int,
    hidden_level: int,
    adapter_dir: Path,
    hidden_prompts_by_suite: Mapping[str, Sequence[PromptRecord]],
) -> dict[str, list[CompletionRecord]]:
    q_root = q_bank_root_for(config, split_seed=split_seed, hidden_level=hidden_level)
    records_by_suite: dict[str, list[CompletionRecord]] = {}
    missing_suites: list[str] = []
    for suite in config.prompt_suites:
        path = q_root / "suites" / suite / "completion_bank_q.jsonl"
        expected_prompt_ids = {record.prompt_id for record in hidden_prompts_by_suite[suite]}
        expected_records = config.bank_samples_per_prompt * len(expected_prompt_ids)
        if path.exists() and not config.overwrite_q_banks:
            records = read_completion_records(path)
            actual_prompt_ids = {record.prompt_id for record in records}
            if len(records) == expected_records and actual_prompt_ids == expected_prompt_ids:
                records_by_suite[suite] = records
                continue
        missing_suites.append(suite)

    if missing_suites:
        runtime_config = GenerationRuntimeConfig(
            base_model_id=config.base_model_id,
            adapter_dir=adapter_dir,
            samples_per_prompt=config.bank_samples_per_prompt,
            max_new_tokens=max(section5.SECTION5_SUITE_SPECS[suite].max_new_tokens for suite in config.prompt_suites),
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            num_beams=1,
            do_sample=True,
            dtype=config.eval_dtype,
            device=config.eval_device,
            batch_size=config.q_generation_batch_size,
            prompt_format="raw",
            seed=config.seed,
            progress=config.progress,
            ignore_eos=True,
            backend=config.generation_backend,
            max_num_seqs=config.max_num_seqs,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        with CompletionGenerator(runtime_config) as generator:
            for suite in missing_suites:
                path = q_root / "suites" / suite / "completion_bank_q.jsonl"
                expected_prompt_ids = {record.prompt_id for record in hidden_prompts_by_suite[suite]}
                records = _complete_q_records_from_partial_bank(
                    path,
                    expected_prompt_ids=expected_prompt_ids,
                    samples_per_prompt=config.bank_samples_per_prompt,
                )
                completed_prompt_ids = {record.prompt_id for record in records}
                prompts_to_generate = [
                    record for record in hidden_prompts_by_suite[suite] if record.prompt_id not in completed_prompt_ids
                ]
                path.parent.mkdir(parents=True, exist_ok=True)
                write_completion_records(path, records)
                with path.open("a", encoding="utf-8") as handle:
                    for batch_records in generator.iter_record_batches(
                        suite,
                        prompts_to_generate,
                        model_label="q",
                        adapter_enabled=True,
                        max_new_tokens=section5.SECTION5_SUITE_SPECS[suite].max_new_tokens,
                    ):
                        for record in batch_records:
                            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
                        handle.flush()
                        records.extend(batch_records)
                records_by_suite[suite] = records
    return {suite: records_by_suite[suite] for suite in config.prompt_suites}


def section5_config_for_eval(
    config: PromptConcealmentConfig,
    output_root: Path,
    adapter_dir: Path,
    split_seed: int,
    hidden_level: int,
) -> section5.Section5Config:
    return section5.Section5Config(
        base_model_id=config.base_model_id,
        adapter_dir=adapter_dir,
        output_root=output_root,
        prompt_suites=config.prompt_suites,
        candidate_specs=(
            section5.CandidateSpec(
                label=f"prompt_conceal_split{split_seed:03d}_hidden{hidden_level:03d}",
                model_alias="q",
                adapter_dir=adapter_dir,
            ),
        ),
        bank_samples_per_prompt=config.bank_samples_per_prompt,
        sample_multiplier=config.sample_multiplier,
        n_simulations=config.n_simulations,
        bootstrap_draws=config.bootstrap_draws,
        alpha=config.alpha,
        secondary_alpha=config.secondary_alpha,
        failure_rejection_rate=config.failure_rejection_rate,
        effect_repeats=config.effect_repeats,
        effect_sample_multiplier=config.effect_sample_multiplier,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        dtype=config.eval_dtype,
        device=config.eval_device,
        batch_size=config.q_generation_batch_size,
        generation_backend=config.generation_backend,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=config.gpu_memory_utilization,
        prompt_format="raw",
        seed=config.seed,
        progress=config.progress,
        encoding="token",
    )


def load_prompt_records_by_suite(
    root: Path,
    *,
    prompt_suites: Sequence[str] = DEFAULT_PROMPT_SUITES,
) -> dict[str, list[PromptRecord]]:
    return {
        suite: _read_prompt_records(Path(root) / "suites" / suite / "prompts.jsonl")
        for suite in prompt_suites
    }


def load_p_completion_records_by_suite(
    root: Path,
    *,
    prompt_suites: Sequence[str] = DEFAULT_PROMPT_SUITES,
) -> dict[str, list[CompletionRecord]]:
    return {
        suite: read_completion_records(_reference_completion_bank(Path(root), suite))
        for suite in prompt_suites
    }


def filter_prompt_records_by_suite(
    prompts_by_suite: Mapping[str, Sequence[PromptRecord]],
    prompt_ids_by_suite: Mapping[str, Sequence[str]],
) -> dict[str, list[PromptRecord]]:
    filtered: dict[str, list[PromptRecord]] = {}
    for suite, records in prompts_by_suite.items():
        selected_ids = set(prompt_ids_by_suite.get(suite, []))
        filtered[suite] = [record for record in records if record.prompt_id in selected_ids]
    return filtered


def adapter_run_dir(config: PromptConcealmentConfig, *, split_seed: int, hidden_level: int) -> Path:
    return config.adapter_output_root / f"{config.adapter_run_prefix}_split{split_seed:03d}_hidden{hidden_level:03d}_seed{config.seed}"


def adapter_dir_for(config: PromptConcealmentConfig, *, split_seed: int, hidden_level: int) -> Path:
    return adapter_run_dir(config, split_seed=split_seed, hidden_level=hidden_level) / "adapter"


def split_root_for(config: PromptConcealmentConfig, *, split_seed: int, hidden_level: int) -> Path:
    return config.output_root / "splits" / f"split{split_seed:03d}" / f"hidden{hidden_level:03d}"


def eval_root_for(config: PromptConcealmentConfig, *, split_seed: int, hidden_level: int) -> Path:
    return config.output_root / "evals" / f"split{split_seed:03d}" / f"hidden{hidden_level:03d}"


def q_bank_root_for(config: PromptConcealmentConfig, *, split_seed: int, hidden_level: int) -> Path:
    return eval_root_for(config, split_seed=split_seed, hidden_level=hidden_level) / "q_bank"


def hidden_eval_root_for(config: PromptConcealmentConfig, *, split_seed: int, hidden_level: int) -> Path:
    return eval_root_for(config, split_seed=split_seed, hidden_level=hidden_level) / "hidden"


LONG_FIELDNAMES = [
    "hidden_level",
    "hidden_fraction",
    "split_seed",
    "suite",
    "prompt_count",
    "sample_size_per_side",
    "rejection_rate_alpha_0_05",
    "fail_alpha_0_05",
    "pvalue_mean",
    "mmd_mean",
    "aggregate_reject",
    "adapter_dir",
    "eval_dir",
    "row_type",
]

AGGREGATE_FIELDNAMES = [
    "hidden_level",
    "hidden_fraction",
    "suite",
    "seed_count",
    "mean_rejection_rate_alpha_0_05",
    "std_rejection_rate_alpha_0_05",
    "min_rejection_rate_alpha_0_05",
    "max_rejection_rate_alpha_0_05",
    "mean_pvalue",
    "mean_mmd",
    "fail_rate",
    "row_type",
]

DECISION_FIELDNAMES = [
    "hidden_level",
    "hidden_fraction",
    "seed_count",
    "mean_aggregate_reject_rate",
    "std_aggregate_reject_rate",
    "true_count",
    "false_count",
    "row_type",
]

COMPARISON_FIELDNAMES = [
    "hidden_level",
    "suite",
    "baseline_seed_count",
    "current_seed_count",
    "baseline_mean_rejection_rate_alpha_0_05",
    "current_mean_rejection_rate_alpha_0_05",
    "delta_mean_rejection_rate_alpha_0_05",
    "baseline_std_rejection_rate_alpha_0_05",
    "current_std_rejection_rate_alpha_0_05",
    "baseline_fail_rate",
    "current_fail_rate",
    "delta_fail_rate",
]

COMPARISON_DECISION_FIELDNAMES = [
    "hidden_level",
    "baseline_seed_count",
    "current_seed_count",
    "baseline_aggregate_reject_rate",
    "current_aggregate_reject_rate",
    "delta_aggregate_reject_rate",
    "baseline_true_count",
    "current_true_count",
    "baseline_false_count",
    "current_false_count",
]


def _load_per_seed_summary_rows(config: PromptConcealmentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_seed in config.split_seeds:
        for hidden_level in config.hidden_levels:
            manifest_path = split_root_for(config, split_seed=split_seed, hidden_level=hidden_level) / "split_manifest.json"
            summary_path = hidden_eval_root_for(config, split_seed=split_seed, hidden_level=hidden_level) / "summary.json"
            if not manifest_path.exists() or not summary_path.exists():
                continue
            manifest = _read_json(manifest_path)
            summary = _read_json(summary_path)
            aggregate_reject = _aggregate_reject(summary)
            for result in summary.get("results", []):
                if result.get("suite") not in config.prompt_suites:
                    continue
                rows.append(
                    {
                        "hidden_level": int(hidden_level),
                        "hidden_fraction": float(manifest.get("hidden_fraction_actual", int(hidden_level) / 100.0)),
                        "split_seed": int(split_seed),
                        "suite": result.get("suite", ""),
                        "prompt_count": result.get("prompts", ""),
                        "sample_size_per_side": result.get("sample_size_per_side", ""),
                        "rejection_rate_alpha_0_05": _result_rejection_rate_005(result),
                        "fail_alpha_0_05": _result_fail_005(result, config.failure_rejection_rate),
                        "pvalue_mean": result.get("mean_pvalue", ""),
                        "mmd_mean": result.get("mean_mmd", ""),
                        "aggregate_reject": aggregate_reject,
                        "adapter_dir": str(adapter_dir_for(config, split_seed=split_seed, hidden_level=hidden_level)),
                        "eval_dir": str(summary_path.parent),
                        "row_type": "split",
                    }
                )
    return rows


def write_trace_ablation_comparison(
    config: PromptConcealmentConfig,
    *,
    baseline_root: Path = DEFAULT_BASELINE_FRONTIER_ROOT,
    image_dir: Path | None = Path("images"),
) -> dict[str, Any]:
    baseline_long_path = baseline_root / "summary_long.csv"
    current_long_path = config.output_root / "summary_long.csv"
    if not baseline_long_path.exists() or not current_long_path.exists():
        payload = {
            "status": "not_run",
            "reason": "missing_summary_long",
            "baseline_summary_long": str(baseline_long_path),
            "current_summary_long": str(current_long_path),
        }
        _write_json(config.output_root / "comparison_summary.json", payload)
        return payload

    seeds = {str(seed) for seed in config.split_seeds}
    hidden_levels = {int(level) for level in config.hidden_levels}
    baseline_rows = _filter_comparison_rows(_read_csv_dicts(baseline_long_path), seeds, hidden_levels, config.prompt_suites)
    current_rows = _filter_comparison_rows(_read_csv_dicts(current_long_path), seeds, hidden_levels, config.prompt_suites)
    baseline_aggregate = _aggregate_suite_rows(baseline_rows)
    current_aggregate = _aggregate_suite_rows(current_rows)
    comparison_rows = _build_suite_comparison_rows(baseline_aggregate, current_aggregate)
    baseline_decisions = _aggregate_decision_rows(baseline_rows, ())
    current_decisions = _aggregate_decision_rows(current_rows, ())
    decision_rows = _build_decision_comparison_rows(baseline_decisions, current_decisions)

    _write_csv_with_fields(config.output_root / "comparison_summary.csv", comparison_rows, COMPARISON_FIELDNAMES)
    _write_csv_with_fields(
        config.output_root / "comparison_decision_summary.csv",
        decision_rows,
        COMPARISON_DECISION_FIELDNAMES,
    )
    figure_path = ""
    if image_dir is not None:
        figure_path = _write_trace_ablation_plot(comparison_rows, image_dir, figure_prefix=config.figure_prefix)
    payload = {
        "status": "complete",
        "baseline_root": str(baseline_root),
        "current_root": str(config.output_root),
        "split_seeds": list(config.split_seeds),
        "hidden_levels": list(config.hidden_levels),
        "suite_rows": comparison_rows,
        "decision_rows": decision_rows,
        "figure_path": figure_path,
    }
    _write_json(config.output_root / "comparison_summary.json", payload)
    return payload


def _filter_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    split_seeds: set[str],
    hidden_levels: set[int],
    prompt_suites: Sequence[str],
) -> list[dict[str, Any]]:
    suites = set(prompt_suites)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("row_type", "")) != "split":
            continue
        if str(row.get("split_seed", "")) not in split_seeds:
            continue
        if int(row["hidden_level"]) not in hidden_levels:
            continue
        if str(row.get("suite", "")) not in suites:
            continue
        filtered.append(dict(row))
    return filtered


def _build_suite_comparison_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_key = {(int(row["hidden_level"]), str(row["suite"])): row for row in baseline_rows}
    current_by_key = {(int(row["hidden_level"]), str(row["suite"])): row for row in current_rows}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(baseline_by_key) & set(current_by_key)):
        baseline = baseline_by_key[key]
        current = current_by_key[key]
        baseline_rate = float(baseline["mean_rejection_rate_alpha_0_05"])
        current_rate = float(current["mean_rejection_rate_alpha_0_05"])
        baseline_fail = float(baseline["fail_rate"])
        current_fail = float(current["fail_rate"])
        rows.append(
            {
                "hidden_level": key[0],
                "suite": key[1],
                "baseline_seed_count": baseline["seed_count"],
                "current_seed_count": current["seed_count"],
                "baseline_mean_rejection_rate_alpha_0_05": baseline_rate,
                "current_mean_rejection_rate_alpha_0_05": current_rate,
                "delta_mean_rejection_rate_alpha_0_05": current_rate - baseline_rate,
                "baseline_std_rejection_rate_alpha_0_05": baseline["std_rejection_rate_alpha_0_05"],
                "current_std_rejection_rate_alpha_0_05": current["std_rejection_rate_alpha_0_05"],
                "baseline_fail_rate": baseline_fail,
                "current_fail_rate": current_fail,
                "delta_fail_rate": current_fail - baseline_fail,
            }
        )
    return rows


def _build_decision_comparison_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_level = {int(row["hidden_level"]): row for row in baseline_rows}
    current_by_level = {int(row["hidden_level"]): row for row in current_rows}
    rows: list[dict[str, Any]] = []
    for hidden_level in sorted(set(baseline_by_level) & set(current_by_level)):
        baseline = baseline_by_level[hidden_level]
        current = current_by_level[hidden_level]
        baseline_rate = float(baseline["mean_aggregate_reject_rate"])
        current_rate = float(current["mean_aggregate_reject_rate"])
        rows.append(
            {
                "hidden_level": hidden_level,
                "baseline_seed_count": baseline["seed_count"],
                "current_seed_count": current["seed_count"],
                "baseline_aggregate_reject_rate": baseline_rate,
                "current_aggregate_reject_rate": current_rate,
                "delta_aggregate_reject_rate": current_rate - baseline_rate,
                "baseline_true_count": baseline["true_count"],
                "current_true_count": current["true_count"],
                "baseline_false_count": baseline["false_count"],
                "current_false_count": current["false_count"],
            }
        )
    return rows


def _load_endpoint_rows(
    *,
    endpoint_u40_summary: Path,
    endpoint_poisoned_summary: Path,
    prompt_suites: Sequence[str],
    failure_rejection_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_specs = [
        (0, endpoint_u40_summary, None, "endpoint_u40"),
        (100, endpoint_poisoned_summary, "poisoned", "endpoint_poisoned"),
    ]
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for hidden_level, summary_path, candidate, row_type in endpoint_specs:
        if not Path(summary_path).exists():
            continue
        payload = _read_json(Path(summary_path))
        suite_rows: list[dict[str, Any]] = []
        for result in payload.get("results", []):
            if candidate is not None and result.get("candidate") != candidate:
                continue
            suite = str(result.get("suite", ""))
            if suite not in prompt_suites:
                continue
            rejection_rate = _result_rejection_rate_005(result)
            fail = _result_fail_005(result, failure_rejection_rate)
            suite_rows.append(
                {
                    "hidden_level": int(hidden_level),
                    "hidden_fraction": float(hidden_level) / 100.0,
                    "split_seed": "endpoint",
                    "suite": suite,
                    "prompt_count": result.get("prompts", ""),
                    "sample_size_per_side": result.get("sample_size_per_side", ""),
                    "rejection_rate_alpha_0_05": rejection_rate,
                    "fail_alpha_0_05": fail,
                    "pvalue_mean": result.get("mean_pvalue", ""),
                    "mmd_mean": result.get("mean_mmd", ""),
                    "aggregate_reject": "",
                    "adapter_dir": "",
                    "eval_dir": str(Path(summary_path).parent),
                    "row_type": row_type,
                }
            )
        aggregate_reject = any(bool(row["fail_alpha_0_05"]) for row in suite_rows)
        for row in suite_rows:
            row["aggregate_reject"] = aggregate_reject
        rows.extend(suite_rows)
        if suite_rows:
            decisions.append(
                {
                    "hidden_level": int(hidden_level),
                    "hidden_fraction": float(hidden_level) / 100.0,
                    "aggregate_reject": aggregate_reject,
                    "row_type": row_type,
                }
            )
    return rows, decisions


def _aggregate_suite_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["hidden_level"]), str(row["suite"])), []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (hidden_level, suite), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        rates = [float(row["rejection_rate_alpha_0_05"]) for row in group]
        pvalues = [_float_or_none(row.get("pvalue_mean")) for row in group]
        mmds = [_float_or_none(row.get("mmd_mean")) for row in group]
        fail_values = [_bool_from_cell(row.get("fail_alpha_0_05")) for row in group]
        aggregate_rows.append(
            {
                "hidden_level": hidden_level,
                "hidden_fraction": statistics.mean(float(row["hidden_fraction"]) for row in group),
                "suite": suite,
                "seed_count": len(group),
                "mean_rejection_rate_alpha_0_05": statistics.mean(rates),
                "std_rejection_rate_alpha_0_05": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                "min_rejection_rate_alpha_0_05": min(rates),
                "max_rejection_rate_alpha_0_05": max(rates),
                "mean_pvalue": _mean_defined(pvalues),
                "mean_mmd": _mean_defined(mmds),
                "fail_rate": sum(1 for value in fail_values if value) / len(fail_values),
                "row_type": str(group[0].get("row_type", "")),
            }
        )
    return aggregate_rows


def _aggregate_decision_rows(
    per_seed_rows: Sequence[Mapping[str, Any]],
    endpoint_decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[Any, Mapping[str, Any]]] = {}
    for row in per_seed_rows:
        hidden_level = int(row["hidden_level"])
        split_seed = row["split_seed"]
        grouped.setdefault(hidden_level, {})[split_seed] = row
    decision_rows: list[dict[str, Any]] = []
    for hidden_level, by_seed in sorted(grouped.items()):
        values = [1.0 if _bool_from_cell(row.get("aggregate_reject")) else 0.0 for row in by_seed.values()]
        hidden_fraction = statistics.mean(float(row["hidden_fraction"]) for row in by_seed.values())
        decision_rows.append(
            {
                "hidden_level": hidden_level,
                "hidden_fraction": hidden_fraction,
                "seed_count": len(values),
                "mean_aggregate_reject_rate": statistics.mean(values),
                "std_aggregate_reject_rate": statistics.stdev(values) if len(values) > 1 else 0.0,
                "true_count": int(sum(values)),
                "false_count": len(values) - int(sum(values)),
                "row_type": "split",
            }
        )
    for endpoint in endpoint_decisions:
        value = 1.0 if _bool_from_cell(endpoint.get("aggregate_reject")) else 0.0
        decision_rows.append(
            {
                "hidden_level": int(endpoint["hidden_level"]),
                "hidden_fraction": float(endpoint["hidden_fraction"]),
                "seed_count": 1,
                "mean_aggregate_reject_rate": value,
                "std_aggregate_reject_rate": 0.0,
                "true_count": int(value),
                "false_count": 1 - int(value),
                "row_type": endpoint.get("row_type", "endpoint"),
            }
        )
    return sorted(decision_rows, key=lambda row: int(row["hidden_level"]))


def _compute_thresholds(
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_suites: Sequence[str],
) -> dict[str, int | None]:
    by_level: dict[int, dict[str, float]] = {}
    for row in aggregate_rows:
        by_level.setdefault(int(row["hidden_level"]), {})[str(row["suite"])] = float(
            row["mean_rejection_rate_alpha_0_05"]
        )
    any_threshold = None
    all_threshold = None
    for hidden_level in sorted(by_level):
        suite_rates = [by_level[hidden_level].get(suite) for suite in prompt_suites]
        defined_rates = [rate for rate in suite_rates if rate is not None]
        if any_threshold is None and any(rate >= 0.5 for rate in defined_rates):
            any_threshold = hidden_level
        if (
            all_threshold is None
            and len(defined_rates) == len(tuple(prompt_suites))
            and all(rate >= 0.5 for rate in defined_rates)
        ):
            all_threshold = hidden_level
    return {
        "any_suite_mean_ge_0_5": any_threshold,
        "all_suites_mean_ge_0_5": all_threshold,
    }


def _write_rejection_plot(rows: Sequence[Mapping[str, Any]], image_dir: Path, *, figure_prefix: str = DEFAULT_FIGURE_PREFIX) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{figure_prefix}_rejection_rates.png"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    suites = sorted({str(row["suite"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for suite in suites:
        suite_rows = sorted([row for row in rows if row["suite"] == suite], key=lambda row: int(row["hidden_level"]))
        xs = [int(row["hidden_level"]) for row in suite_rows]
        ys = [float(row["mean_rejection_rate_alpha_0_05"]) for row in suite_rows]
        yerr = [float(row["std_rejection_rate_alpha_0_05"]) for row in suite_rows]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, linewidth=1.8, label=suite)
    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prompt distribution hidden from model owner (%)")
    ax.set_ylabel("MET rejection rate at alpha=0.05")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _write_decision_plot(rows: Sequence[Mapping[str, Any]], image_dir: Path, *, figure_prefix: str = DEFAULT_FIGURE_PREFIX) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{figure_prefix}_decision_rates.png"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    ordered = sorted(rows, key=lambda row: int(row["hidden_level"]))
    xs = [int(row["hidden_level"]) for row in ordered]
    ys = [float(row["mean_aggregate_reject_rate"]) for row in ordered]
    yerr = [float(row["std_aggregate_reject_rate"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(xs, ys, width=12, color="#0072B2", alpha=0.86)
    ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    ax.set_xlabel("Prompt distribution hidden from model owner (%)")
    ax.set_ylabel("Aggregate MET reject rate")
    ax.set_ylim(0.0, 1.1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _write_trace_ablation_plot(
    rows: Sequence[Mapping[str, Any]],
    image_dir: Path,
    *,
    figure_prefix: str,
) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{figure_prefix}_trace_ablation.png"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    ordered = sorted(rows, key=lambda row: (int(row["hidden_level"]), str(row["suite"])))
    if not ordered:
        return ""
    labels = [f"{row['suite']}\nhidden{int(row['hidden_level']):03d}" for row in ordered]
    baseline = [float(row["baseline_mean_rejection_rate_alpha_0_05"]) for row in ordered]
    current = [float(row["current_mean_rejection_rate_alpha_0_05"]) for row in ordered]
    xs = list(range(len(ordered)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.8, len(xs) * 1.4), 4.4))
    ax.bar([x - width / 2 for x in xs], baseline, width=width, label="u40 baseline", color="#999999")
    ax.bar([x + width / 2 for x in xs], current, width=width, label="w80/h80/u160", color="#0072B2")
    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MET rejection rate at alpha=0.05")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _read_prompt_records(path: Path) -> list[PromptRecord]:
    return [
        PromptRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reference_completion_bank(root: Path, suite: str) -> Path:
    suite_dir = root / "suites" / suite
    for filename in REFERENCE_BANK_FILENAMES:
        path = suite_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"No reference P completion bank found under {suite_dir}")


def _hidden_prompt_count(prompt_count: int, hidden_level: int) -> int:
    if hidden_level <= 0:
        return 0
    if hidden_level >= 100:
        return prompt_count
    return int(math.floor(prompt_count * hidden_level / 100.0))


def _parse_selected_trace_counts(
    values: Sequence[str],
    prompt_suites: Sequence[str],
) -> dict[str, int] | None:
    if not values:
        return None
    suite_set = set(prompt_suites)
    parsed: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --selected-trace-count SUITE=COUNT, got {value!r}")
        suite, count_text = value.split("=", 1)
        suite = suite.strip()
        if suite not in suite_set:
            raise ValueError(f"Unknown suite {suite!r}; expected one of {sorted(suite_set)}")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"Trace count for {suite!r} must be an integer, got {count_text!r}") from exc
        if count <= 0:
            raise ValueError(f"Trace count for {suite!r} must be positive, got {count}")
        parsed[suite] = count
    missing = suite_set - set(parsed)
    if missing:
        raise ValueError(f"Missing --selected-trace-count for suites: {sorted(missing)}")
    return dict(sorted(parsed.items()))


def _sha256_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def _filter_jsonl_by_prompt_ids(input_path: Path, output_path: Path, prompt_ids: set[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("prompt_id")) in prompt_ids:
                sink.write(line)


def _filter_completion_jsonl_by_suite_prompt_ids(
    input_path: Path,
    output_path: Path,
    prompt_ids_by_suite: Mapping[str, set[str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            suite = str(row.get("suite"))
            if str(row.get("prompt_id")) in prompt_ids_by_suite.get(suite, set()):
                sink.write(line)


def _sort_prompt_ids_by_reference_order(
    prompt_ids_by_suite: Mapping[str, Sequence[str]],
    all_prompt_ids_by_suite: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    sorted_by_suite: dict[str, list[str]] = {}
    for suite, prompt_ids in prompt_ids_by_suite.items():
        selected = set(prompt_ids)
        sorted_by_suite[suite] = [prompt_id for prompt_id in all_prompt_ids_by_suite[suite] if prompt_id in selected]
    return sorted_by_suite


def _complete_q_records_from_partial_bank(
    path: Path,
    *,
    expected_prompt_ids: set[str],
    samples_per_prompt: int,
) -> list[CompletionRecord]:
    if not path.exists():
        return []
    try:
        records = read_completion_records(path)
    except Exception:
        return []
    grouped: dict[str, list[CompletionRecord]] = {}
    for record in records:
        if record.prompt_id in expected_prompt_ids:
            grouped.setdefault(record.prompt_id, []).append(record)
    complete_records: list[CompletionRecord] = []
    expected_indices = set(range(samples_per_prompt))
    for prompt_id in sorted(grouped):
        by_index = {record.sample_index: record for record in grouped[prompt_id]}
        if set(by_index) == expected_indices:
            complete_records.extend(by_index[index] for index in sorted(by_index))
    return complete_records


def _result_rejection_rate_005(result: Mapping[str, Any]) -> float:
    if "rejection_rate_alpha_0_05" in result:
        return float(result["rejection_rate_alpha_0_05"])
    if "rejection_rate" in result:
        return float(result["rejection_rate"])
    rates = result.get("rejection_rate_by_alpha", {})
    if isinstance(rates, Mapping) and "0.05" in rates:
        return float(rates["0.05"])
    raise KeyError(f"Result has no alpha=0.05 rejection rate: {result}")


def _result_fail_005(result: Mapping[str, Any], failure_rejection_rate: float) -> bool:
    if "fail_alpha_0_05" in result:
        return bool(result["fail_alpha_0_05"])
    return _result_rejection_rate_005(result) >= failure_rejection_rate


def _aggregate_reject(summary: Mapping[str, Any]) -> bool:
    aggregate = summary.get("aggregate", {})
    if isinstance(aggregate, Mapping):
        reject_by_alpha = aggregate.get("reject_by_alpha", {})
        if isinstance(reject_by_alpha, Mapping) and "0.05" in reject_by_alpha:
            return bool(reject_by_alpha["0.05"])
        if "reject" in aggregate:
            return bool(aggregate["reject"])
    return any(_bool_from_cell(result.get("fail_alpha_0_05", result.get("fail"))) for result in summary.get("results", []))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mean_defined(values: Sequence[float | None]) -> float | str:
    defined = [float(value) for value in values if value is not None]
    return statistics.mean(defined) if defined else ""


def _bool_from_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _copy_json(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv_with_fields(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_readme(output_root: Path, config: PromptConcealmentConfig, figure_path: str, decision_figure_path: str) -> None:
    if config.selected_trace_counts_by_suite is not None:
        trace_counts = config.selected_trace_counts_by_suite
    elif (config.selected_train_root / "split_manifest.json").exists():
        trace_counts = _read_json(config.selected_train_root / "split_manifest.json").get(
            "traces_per_prompt_by_suite",
            DEFAULT_TRACE_COUNTS_BY_SUITE,
        )
    else:
        trace_counts = DEFAULT_TRACE_COUNTS_BY_SUITE
    lines = [
        "# Section 6.3 Prompt-Concealment MET Frontier",
        "",
        "This artifact trains KL-tail adapters on visible MET prompts only and evaluates API-faithful MET on hidden prompts.",
        "",
        "## Key Settings",
        "",
        f"- Split seeds: {', '.join(str(seed) for seed in config.split_seeds)}",
        f"- Hidden levels: {', '.join(str(level) for level in config.hidden_levels)}",
        (
            "- Training traces per visible prompt: "
            f"wikipedia_en={trace_counts.get('wikipedia_en')}, "
            f"humaneval={trace_counts.get('humaneval')}, "
            f"ultrachat={trace_counts.get('ultrachat')}"
        ),
        f"- KL steps: {config.max_steps}",
        f"- MET alpha: {config.alpha}",
        f"- MET simulations: {config.n_simulations}",
        f"- Bootstrap draws: {config.bootstrap_draws}",
        "",
        "## Outputs",
        "",
        "- `summary_long.csv`: per-seed, per-suite rejection rates plus endpoints.",
        "- `summary.csv`: mean/std suite rejection rates by hidden level.",
        "- `decision_summary.csv`: aggregate reject rates by hidden level.",
        "- `comparison_summary.csv`: suite-level trace ablation against the u40 baseline when available.",
        "- `comparison_decision_summary.csv`: aggregate trace ablation against the u40 baseline when available.",
        f"- Rejection-rate figure: `{figure_path}`",
        f"- Decision-rate figure: `{decision_figure_path}`",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = PromptConcealmentConfig.from_args(args)
    result = run_pipeline(
        config,
        phase=args.phase,
        endpoint_u40_summary=args.endpoint_u40_summary,
        endpoint_poisoned_summary=args.endpoint_poisoned_summary,
        image_dir=args.image_dir,
    )
    print(json.dumps({"phase": args.phase, "output_root": str(config.output_root), "status": "complete"}, sort_keys=True))
    if args.phase == "summarize":
        print(json.dumps({"thresholds": result.get("thresholds", {})}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
