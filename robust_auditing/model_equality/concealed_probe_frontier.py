from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from robust_auditing.model_equality import section5
from robust_auditing.model_equality.completions import CompletionRecord, read_completion_records, write_completion_records
from robust_auditing.model_equality.constants import DEFAULT_MODEL_ID
from robust_auditing.model_equality.generation import CompletionGenerator, GenerationRuntimeConfig
from robust_auditing.model_equality.prompts import PromptRecord, write_prompt_records


DEFAULT_ADAPTER_DIR = Path("outputs/targeted_ft/fullsuite_api_met_kl_s150_w20_h20_u40_from_exact_chain_seed0/adapter")
DEFAULT_PUBLIC_ANCHOR_ROOT = Path(
    "artifacts/model_equality_section5/api_kl_tail_search_ultrachat_k3040_seed0/evals/"
    "api_met_kl_s150_w20_h20_u40"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/model_equality_section5/concealed_probe_frontier_api_met_kl_s150_w20_h20_u40_seed0_9"
)
DEFAULT_PROMPT_SUITES = ("wikipedia_en", "humaneval", "ultrachat")
DEFAULT_PUBLIC_COUNTS = {"wikipedia_en": 25, "humaneval": 20, "ultrachat": 20}
DEFAULT_CONCEALED_POOL_COUNTS = {"wikipedia_en": 50, "humaneval": 40, "ultrachat": 40}
DEFAULT_CONCEALED_LEVELS = (25, 50, 75, 100)
DEFAULT_SPLIT_SEEDS = tuple(range(10))
DEFAULT_FIGURE_PREFIX = "met_concealed_probe"


@dataclass(frozen=True)
class ConcealedPromptPool:
    public_counts_by_suite: dict[str, int]
    concealed_pool_counts_by_suite: dict[str, int]
    concealed_prompts_by_suite: dict[str, list[PromptRecord]]

    @property
    def prompt_count_by_suite(self) -> dict[str, int]:
        return {suite: len(records) for suite, records in self.concealed_prompts_by_suite.items()}

    @property
    def total_prompt_count(self) -> int:
        return sum(len(records) for records in self.concealed_prompts_by_suite.values())


@dataclass(frozen=True)
class ConcealedProbeSplit:
    split_seed: int
    concealed_level: int
    prompt_order_by_suite: dict[str, list[str]]
    concealed_prompt_ids_by_suite: dict[str, list[str]]
    pool_prompt_ids_by_suite: dict[str, list[str]]

    @property
    def concealed_prompt_count(self) -> int:
        return sum(len(ids) for ids in self.concealed_prompt_ids_by_suite.values())


@dataclass(frozen=True)
class ConcealedProbeConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    public_anchor_root: Path = DEFAULT_PUBLIC_ANCHOR_ROOT
    adapter_dir: Path = DEFAULT_ADAPTER_DIR
    base_model_id: str = DEFAULT_MODEL_ID
    met_repo_root: Path = section5.DEFAULT_MET_REPO_ROOT
    prompt_suites: tuple[str, ...] = DEFAULT_PROMPT_SUITES
    public_counts_by_suite: dict[str, int] | None = None
    concealed_pool_counts_by_suite: dict[str, int] | None = None
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS
    concealed_levels: tuple[int, ...] = DEFAULT_CONCEALED_LEVELS
    bank_samples_per_prompt: int = 250
    sample_multiplier: int = 10
    n_simulations: int = 100
    bootstrap_draws: int = 1000
    alpha: float = 0.05
    secondary_alpha: float = 0.05
    failure_rejection_rate: float = 0.5
    effect_repeats: int = 10
    effect_sample_multiplier: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = 0
    dtype: str = "bf16"
    device: str = "cuda"
    batch_size: int = 64
    generation_backend: str = "hf"
    max_num_seqs: int = 1024
    gpu_memory_utilization: float = 0.95
    prompt_format: str = "raw"
    seed: int = 0
    progress: bool = True
    overwrite_banks: bool = False
    overwrite_eval: bool = False
    figure_prefix: str = DEFAULT_FIGURE_PREFIX

    @property
    def resolved_public_counts_by_suite(self) -> dict[str, int]:
        return _counts_for_suites(self.public_counts_by_suite or DEFAULT_PUBLIC_COUNTS, self.prompt_suites)

    @property
    def resolved_concealed_pool_counts_by_suite(self) -> dict[str, int]:
        return _counts_for_suites(self.concealed_pool_counts_by_suite or DEFAULT_CONCEALED_POOL_COUNTS, self.prompt_suites)

    @property
    def public_anchor_summary(self) -> Path:
        return self.public_anchor_root / "summary.json"

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("output_root", "public_anchor_root", "adapter_dir", "met_repo_root"):
            payload[key] = str(payload[key])
        payload["public_counts_by_suite"] = self.resolved_public_counts_by_suite
        payload["concealed_pool_counts_by_suite"] = self.resolved_concealed_pool_counts_by_suite
        return payload

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ConcealedProbeConfig":
        prompt_suites = tuple(args.prompt_suite) if args.prompt_suite else DEFAULT_PROMPT_SUITES
        return cls(
            output_root=Path(args.output_root),
            public_anchor_root=Path(args.public_anchor_root),
            adapter_dir=Path(args.adapter_dir),
            base_model_id=args.base_model_id,
            met_repo_root=Path(args.met_repo_root),
            prompt_suites=prompt_suites,
            split_seeds=tuple(args.split_seed) if args.split_seed else DEFAULT_SPLIT_SEEDS,
            concealed_levels=tuple(args.concealed_level) if args.concealed_level else DEFAULT_CONCEALED_LEVELS,
            bank_samples_per_prompt=args.bank_samples_per_prompt,
            sample_multiplier=args.sample_multiplier,
            n_simulations=args.n_simulations,
            bootstrap_draws=args.bootstrap_draws,
            alpha=args.alpha,
            secondary_alpha=args.secondary_alpha,
            failure_rejection_rate=args.failure_rejection_rate,
            effect_repeats=args.effect_repeats,
            effect_sample_multiplier=args.effect_sample_multiplier,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            dtype=args.dtype,
            device=args.device,
            batch_size=args.batch_size,
            generation_backend=args.generation_backend,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prompt_format=args.prompt_format,
            seed=args.seed,
            progress=not args.no_progress,
            overwrite_banks=args.overwrite_banks,
            overwrite_eval=args.overwrite_eval,
            figure_prefix=args.figure_prefix,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-adapter concealed-probe Section 6.3 MET frontier.")
    parser.add_argument(
        "--phase",
        choices=("prepare", "generate-banks", "evaluate", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--public-anchor-root", type=Path, default=DEFAULT_PUBLIC_ANCHOR_ROOT)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--met-repo-root", type=Path, default=section5.DEFAULT_MET_REPO_ROOT)
    parser.add_argument("--prompt-suite", action="append", choices=tuple(section5.SECTION5_SUITE_SPECS))
    parser.add_argument("--split-seed", action="append", type=int)
    parser.add_argument("--concealed-level", action="append", type=int)
    parser.add_argument("--bank-samples-per-prompt", type=int, default=250)
    parser.add_argument("--sample-multiplier", type=int, default=10)
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--secondary-alpha", type=float, default=0.05)
    parser.add_argument("--failure-rejection-rate", type=float, default=0.5)
    parser.add_argument("--effect-repeats", type=int, default=10)
    parser.add_argument("--effect-sample-multiplier", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--generation-backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--prompt-format", choices=("raw", "chat", "auto"), default="raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figure-prefix", default=DEFAULT_FIGURE_PREFIX)
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--overwrite-banks", action="store_true")
    parser.add_argument("--overwrite-eval", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def build_concealed_prompt_pool_from_records(
    records_by_suite: Mapping[str, Sequence[PromptRecord]],
    *,
    public_counts_by_suite: Mapping[str, int],
    concealed_pool_counts_by_suite: Mapping[str, int],
    public_anchor_root: Path | None = None,
) -> ConcealedPromptPool:
    concealed_by_suite: dict[str, list[PromptRecord]] = {}
    for suite, public_count in public_counts_by_suite.items():
        pool_count = int(concealed_pool_counts_by_suite[suite])
        records = list(records_by_suite[suite])
        required = int(public_count) + pool_count
        if len(records) < required:
            raise ValueError(f"{suite} materialised {len(records)} prompts, expected at least {required}")
        if public_anchor_root is not None:
            _assert_public_prefix_matches_anchor(
                suite,
                records[: int(public_count)],
                Path(public_anchor_root) / "suites" / suite / "prompts.jsonl",
            )
        concealed_records = []
        for local_index, record in enumerate(records[int(public_count) : required]):
            stream_index = int(public_count) + local_index
            metadata = dict(record.metadata)
            metadata.setdefault("original_prompt_id", record.prompt_id)
            metadata["stream_index"] = stream_index
            concealed_records.append(
                PromptRecord(
                    suite=suite,
                    prompt_id=str(stream_index),
                    text=record.text,
                    metadata=metadata,
                )
            )
        public_ids = {str(record.prompt_id) for record in records[: int(public_count)]}
        concealed_ids = {record.prompt_id for record in concealed_records}
        if public_ids & concealed_ids:
            raise ValueError(f"{suite} concealed pool is not disjoint from public prompt IDs")
        concealed_by_suite[suite] = concealed_records
    return ConcealedPromptPool(
        public_counts_by_suite={suite: int(count) for suite, count in public_counts_by_suite.items()},
        concealed_pool_counts_by_suite={suite: int(count) for suite, count in concealed_pool_counts_by_suite.items()},
        concealed_prompts_by_suite=concealed_by_suite,
    )


def load_concealed_prompt_pool_from_upstream(config: ConcealedProbeConfig) -> ConcealedPromptPool:
    prompt_module = section5.import_upstream_prompts_module(
        section5.Section5Config(
            base_model_id=config.base_model_id,
            met_repo_root=config.met_repo_root,
            prompt_suites=config.prompt_suites,
        )
    )
    formatter = section5.HuggingFaceChatFormatter(config.base_model_id)
    records_by_suite: dict[str, list[PromptRecord]] = {}
    public_counts = config.resolved_public_counts_by_suite
    pool_counts = config.resolved_concealed_pool_counts_by_suite
    for suite in config.prompt_suites:
        spec = section5.SECTION5_SUITE_SPECS[suite]
        getter = getattr(prompt_module, spec.getter_name)
        total = public_counts[suite] + pool_counts[suite]
        rows = section5._take_dataset_rows(getter(formatter), total)
        records_by_suite[suite] = [
            section5._prompt_record_from_upstream_row(spec, index, row)
            for index, row in enumerate(rows)
        ]
    return build_concealed_prompt_pool_from_records(
        records_by_suite,
        public_counts_by_suite=public_counts,
        concealed_pool_counts_by_suite=pool_counts,
        public_anchor_root=config.public_anchor_root,
    )


def write_concealed_pool(config: ConcealedProbeConfig, pool: ConcealedPromptPool) -> dict[str, Any]:
    root = concealed_pool_root(config)
    for suite, records in pool.concealed_prompts_by_suite.items():
        write_prompt_records(root / "suites" / suite / "prompts.jsonl", records)
    manifest = {
        "public_counts_by_suite": pool.public_counts_by_suite,
        "concealed_pool_counts_by_suite": pool.concealed_pool_counts_by_suite,
        "concealed_prompt_count_by_suite": pool.prompt_count_by_suite,
        "total_concealed_prompt_count": pool.total_prompt_count,
        "prompt_ids_by_suite": {
            suite: [record.prompt_id for record in records]
            for suite, records in pool.concealed_prompts_by_suite.items()
        },
        "public_anchor_root": str(config.public_anchor_root),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def load_concealed_pool(config: ConcealedProbeConfig) -> ConcealedPromptPool:
    root = concealed_pool_root(config)
    manifest = _read_json(root / "manifest.json")
    prompts_by_suite = {
        suite: _read_prompt_records(root / "suites" / suite / "prompts.jsonl")
        for suite in config.prompt_suites
    }
    return ConcealedPromptPool(
        public_counts_by_suite={suite: int(manifest["public_counts_by_suite"][suite]) for suite in config.prompt_suites},
        concealed_pool_counts_by_suite={
            suite: int(manifest["concealed_pool_counts_by_suite"][suite])
            for suite in config.prompt_suites
        },
        concealed_prompts_by_suite=prompts_by_suite,
    )


def build_concealed_probe_splits(
    pool: ConcealedPromptPool,
    *,
    split_seed: int,
    concealed_levels: Sequence[int],
) -> dict[int, ConcealedProbeSplit]:
    prompt_order_by_suite: dict[str, list[str]] = {}
    pool_prompt_ids_by_suite = {
        suite: [record.prompt_id for record in records]
        for suite, records in pool.concealed_prompts_by_suite.items()
    }
    for suite, prompt_ids in pool_prompt_ids_by_suite.items():
        order = list(prompt_ids)
        random.Random(_sha256_seed(f"{int(split_seed)}\0{suite}")).shuffle(order)
        prompt_order_by_suite[suite] = order

    splits: dict[int, ConcealedProbeSplit] = {}
    for level in concealed_levels:
        ids_by_suite = {}
        for suite, order in prompt_order_by_suite.items():
            count = _concealed_prompt_count(pool.public_counts_by_suite[suite], int(level), len(order))
            ids_by_suite[suite] = list(order[:count])
        splits[int(level)] = ConcealedProbeSplit(
            split_seed=int(split_seed),
            concealed_level=int(level),
            prompt_order_by_suite={suite: list(ids) for suite, ids in prompt_order_by_suite.items()},
            concealed_prompt_ids_by_suite=ids_by_suite,
            pool_prompt_ids_by_suite={suite: list(ids) for suite, ids in pool_prompt_ids_by_suite.items()},
        )
    return splits


def write_concealed_split(config: ConcealedProbeConfig, split: ConcealedProbeSplit) -> dict[str, Any]:
    pool = load_concealed_pool(config)
    root = split_root_for(config, split_seed=split.split_seed, level=split.concealed_level)
    root.mkdir(parents=True, exist_ok=True)
    prompt_lookup = {
        suite: {record.prompt_id: record for record in records}
        for suite, records in pool.concealed_prompts_by_suite.items()
    }
    selected_records = []
    prompt_count_by_suite = {}
    for suite, prompt_ids in split.concealed_prompt_ids_by_suite.items():
        prompt_count_by_suite[suite] = len(prompt_ids)
        selected_records.extend(prompt_lookup[suite][prompt_id] for prompt_id in prompt_ids)
    write_prompt_records(root / "concealed_prompts.jsonl", selected_records)
    manifest = {
        "split_seed": split.split_seed,
        "concealed_level": split.concealed_level,
        "concealed_fraction": split.concealed_level / 100.0,
        "concealed_prompt_count": split.concealed_prompt_count,
        "concealed_prompt_count_by_suite": prompt_count_by_suite,
        "prompt_order_by_suite": split.prompt_order_by_suite,
        "concealed_prompt_ids_by_suite": split.concealed_prompt_ids_by_suite,
        "pool_prompt_ids_by_suite": split.pool_prompt_ids_by_suite,
    }
    _write_json(root / "split_manifest.json", manifest)
    return manifest


def run_prepare_phase(config: ConcealedProbeConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / "config.json", config.to_json())
    pool = load_concealed_prompt_pool_from_upstream(config)
    pool_manifest = write_concealed_pool(config, pool)
    split_manifests: dict[str, Any] = {}
    for split_seed in config.split_seeds:
        splits = build_concealed_probe_splits(pool, split_seed=split_seed, concealed_levels=config.concealed_levels)
        for level in config.concealed_levels:
            manifest = write_concealed_split(config, splits[level])
            split_manifests[f"split{split_seed:03d}/level{level:03d}"] = manifest
    payload = {"config": config.to_json(), "pool": pool_manifest, "splits": split_manifests}
    _write_json(config.output_root / "prepare_summary.json", payload)
    return payload


def ensure_concealed_completion_banks(config: ConcealedProbeConfig) -> dict[str, dict[str, int]]:
    pool = load_concealed_pool(config)
    bank_root = concealed_bank_root(config)
    incomplete: dict[tuple[str, str], list[PromptRecord]] = {}
    for suite, prompt_records in pool.concealed_prompts_by_suite.items():
        expected_ids = {record.prompt_id for record in prompt_records}
        for model_label in ("p", "q"):
            path = bank_path(bank_root, suite, model_label)
            complete_records = [] if config.overwrite_banks else _complete_records_from_partial_bank(
                path,
                expected_prompt_ids=expected_ids,
                samples_per_prompt=config.bank_samples_per_prompt,
                model_label=model_label,
            )
            complete_ids = {record.prompt_id for record in complete_records}
            if len(complete_ids) != len(expected_ids):
                _write_completion_bank(path, complete_records)
                incomplete[(suite, model_label)] = [
                    record for record in prompt_records if record.prompt_id not in complete_ids
                ]
    if incomplete:
        runtime_config = GenerationRuntimeConfig(
            base_model_id=config.base_model_id,
            adapter_dir=config.adapter_dir,
            samples_per_prompt=config.bank_samples_per_prompt,
            max_new_tokens=max(section5.SECTION5_SUITE_SPECS[suite].max_new_tokens for suite in config.prompt_suites),
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_beams=1,
            do_sample=True,
            dtype=config.dtype,
            device=config.device,
            batch_size=config.batch_size,
            prompt_format=config.prompt_format,
            seed=config.seed,
            progress=config.progress,
            ignore_eos=True,
            backend=config.generation_backend,
            max_num_seqs=config.max_num_seqs,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        with CompletionGenerator(runtime_config) as generator:
            for suite in config.prompt_suites:
                for model_label, adapter_enabled in (("p", False), ("q", True)):
                    prompts_to_generate = incomplete.get((suite, model_label), [])
                    if not prompts_to_generate:
                        continue
                    path = bank_path(bank_root, suite, model_label)
                    existing = read_completion_records(path) if path.exists() else []
                    with path.open("a", encoding="utf-8") as handle:
                        for batch_records in generator.iter_record_batches(
                            suite,
                            prompts_to_generate,
                            model_label=model_label,
                            adapter_enabled=adapter_enabled,
                            max_new_tokens=section5.SECTION5_SUITE_SPECS[suite].max_new_tokens,
                        ):
                            for record in batch_records:
                                handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
                            handle.flush()
                            existing.extend(batch_records)
    counts = validate_concealed_completion_banks(
        bank_root,
        pool.concealed_prompts_by_suite,
        samples_per_prompt=config.bank_samples_per_prompt,
    )
    _write_json(config.output_root / "bank_summary.json", counts)
    return counts


def validate_concealed_completion_banks(
    bank_root: Path,
    prompts_by_suite: Mapping[str, Sequence[PromptRecord]],
    *,
    samples_per_prompt: int,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    expected_indices = set(range(int(samples_per_prompt)))
    for suite, prompt_records in prompts_by_suite.items():
        counts[suite] = {}
        expected_prompt_ids = {record.prompt_id for record in prompt_records}
        for model_label in ("p", "q"):
            path = bank_path(Path(bank_root), suite, model_label)
            if not path.exists():
                raise FileNotFoundError(f"Missing {model_label} completion bank for {suite}: {path}")
            records = read_completion_records(path)
            grouped: dict[str, set[int]] = {prompt_id: set() for prompt_id in expected_prompt_ids}
            for record in records:
                if record.prompt_id in grouped and record.model_label == model_label:
                    grouped[record.prompt_id].add(int(record.sample_index))
            for prompt_id, indices in grouped.items():
                if indices != expected_indices:
                    raise ValueError(
                        f"{suite} prompt_id={prompt_id} expected {samples_per_prompt} {model_label} completions "
                        f"with sample indices 0..{samples_per_prompt - 1}, found {len(indices)}"
                    )
            counts[suite][model_label] = len(expected_prompt_ids) * int(samples_per_prompt)
    return counts


def run_evaluate_phase(config: ConcealedProbeConfig) -> dict[str, Any]:
    pool = load_concealed_pool(config)
    validate_concealed_completion_banks(
        concealed_bank_root(config),
        pool.concealed_prompts_by_suite,
        samples_per_prompt=config.bank_samples_per_prompt,
    )
    p_records_by_suite = _load_completion_banks_by_suite(config, "p")
    q_records_by_suite = _load_completion_banks_by_suite(config, "q")
    results: dict[str, Any] = {}
    for split_seed in config.split_seeds:
        for level in config.concealed_levels:
            key = f"split{split_seed:03d}/level{level:03d}"
            manifest = _read_json(split_root_for(config, split_seed=split_seed, level=level) / "split_manifest.json")
            prompt_ids_by_suite = {
                suite: list(manifest["concealed_prompt_ids_by_suite"][suite])
                for suite in config.prompt_suites
            }
            eval_root = eval_root_for(config, split_seed=split_seed, level=level)
            if (eval_root / "summary.json").exists() and not config.overwrite_eval:
                summary = _read_json(eval_root / "summary.json")
            else:
                summary = section5.run_section5_cached_bank_pipeline(
                    section5_config_for_eval(config, eval_root, split_seed, level),
                    prompts_by_suite=pool.concealed_prompts_by_suite,
                    p_records_by_suite=p_records_by_suite,
                    q_records_by_suite=q_records_by_suite,
                    prompt_ids_by_suite=prompt_ids_by_suite,
                )
            results[key] = summary
    _write_json(config.output_root / "eval_summary.json", results)
    return results


def section5_config_for_eval(
    config: ConcealedProbeConfig,
    output_root: Path,
    split_seed: int,
    level: int,
) -> section5.Section5Config:
    return section5.Section5Config(
        base_model_id=config.base_model_id,
        adapter_dir=config.adapter_dir,
        output_root=output_root,
        met_repo_root=config.met_repo_root,
        prompt_suites=config.prompt_suites,
        candidate_specs=(
            section5.CandidateSpec(
                label=f"concealed_probe_split{split_seed:03d}_level{level:03d}",
                model_alias="q",
                adapter_dir=config.adapter_dir,
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
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        dtype=config.dtype,
        device=config.device,
        batch_size=config.batch_size,
        generation_backend=config.generation_backend,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=config.gpu_memory_utilization,
        prompt_format=config.prompt_format,
        seed=config.seed,
        progress=config.progress,
        encoding="token",
    )


def summarize_concealed_probe_results(
    config: ConcealedProbeConfig,
    *,
    public_anchor_summary: Path | None = None,
    image_dir: Path | None = Path("images"),
) -> dict[str, Any]:
    anchor_path = public_anchor_summary if public_anchor_summary is not None else config.public_anchor_summary
    per_seed_rows = _load_per_seed_rows(config)
    anchor_rows, anchor_decision = _load_public_anchor_rows(
        anchor_path,
        prompt_suites=config.prompt_suites,
        failure_rejection_rate=config.failure_rejection_rate,
    )
    long_rows = [*anchor_rows, *per_seed_rows]
    aggregate_rows = _aggregate_suite_rows(long_rows)
    decision_rows = _aggregate_decision_rows(per_seed_rows, anchor_decision)
    thresholds = _compute_thresholds(aggregate_rows, prompt_suites=config.prompt_suites)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_root / "summary_long.csv", long_rows, LONG_FIELDNAMES)
    _write_csv(config.output_root / "summary.csv", aggregate_rows, AGGREGATE_FIELDNAMES)
    _write_csv(config.output_root / "decision_summary.csv", decision_rows, DECISION_FIELDNAMES)
    figure_path = ""
    decision_figure_path = ""
    if image_dir is not None:
        figure_path = _write_rejection_plot(aggregate_rows, image_dir, config.figure_prefix)
        decision_figure_path = _write_decision_plot(decision_rows, image_dir, config.figure_prefix)
    payload = {
        "config": config.to_json(),
        "public_anchor_summary": str(anchor_path),
        "per_seed_rows": per_seed_rows,
        "anchor_rows": anchor_rows,
        "aggregate_rows": aggregate_rows,
        "decision_rows": decision_rows,
        "thresholds": thresholds,
        "figure_path": figure_path,
        "decision_figure_path": decision_figure_path,
    }
    _write_json(config.output_root / "summary.json", payload)
    _write_readme(config.output_root, config, figure_path, decision_figure_path)
    return payload


def run_pipeline(
    config: ConcealedProbeConfig,
    *,
    phase: str,
    image_dir: Path | None = Path("images"),
) -> dict[str, Any]:
    result: dict[str, Any] = {"phase": phase, "output_root": str(config.output_root)}
    if phase in {"prepare", "all"}:
        result = run_prepare_phase(config)
    if phase in {"generate-banks", "all"}:
        result = ensure_concealed_completion_banks(config)
    if phase in {"evaluate", "all"}:
        result = run_evaluate_phase(config)
    if phase in {"summarize", "all"}:
        result = summarize_concealed_probe_results(config, image_dir=image_dir)
    return result


def concealed_pool_root(config: ConcealedProbeConfig) -> Path:
    return config.output_root / "concealed_pool"


def concealed_bank_root(config: ConcealedProbeConfig) -> Path:
    return config.output_root / "banks"


def split_root_for(config: ConcealedProbeConfig, *, split_seed: int, level: int) -> Path:
    return config.output_root / "splits" / f"split{split_seed:03d}" / f"level{level:03d}"


def eval_root_for(config: ConcealedProbeConfig, *, split_seed: int, level: int) -> Path:
    return config.output_root / "evals" / f"split{split_seed:03d}" / f"level{level:03d}"


def bank_path(bank_root: Path, suite: str, model_label: str) -> Path:
    return bank_root / "suites" / suite / f"completion_bank_{model_label}.jsonl"


LONG_FIELDNAMES = [
    "concealed_level",
    "concealed_fraction",
    "split_seed",
    "suite",
    "prompt_count",
    "sample_size_per_side",
    "rejection_rate_alpha_0_05",
    "fail_alpha_0_05",
    "pvalue_mean",
    "mmd_mean",
    "aggregate_reject",
    "eval_dir",
    "row_type",
]

AGGREGATE_FIELDNAMES = [
    "concealed_level",
    "concealed_fraction",
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
    "concealed_level",
    "concealed_fraction",
    "seed_count",
    "mean_aggregate_reject_rate",
    "std_aggregate_reject_rate",
    "true_count",
    "false_count",
    "row_type",
]


def _load_per_seed_rows(config: ConcealedProbeConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_seed in config.split_seeds:
        for level in config.concealed_levels:
            manifest_path = split_root_for(config, split_seed=split_seed, level=level) / "split_manifest.json"
            summary_path = eval_root_for(config, split_seed=split_seed, level=level) / "summary.json"
            if not manifest_path.exists() or not summary_path.exists():
                continue
            manifest = _read_json(manifest_path)
            summary = _read_json(summary_path)
            aggregate_reject = _aggregate_reject(summary)
            for result in summary.get("results", []):
                suite = str(result.get("suite", ""))
                if suite not in config.prompt_suites:
                    continue
                rows.append(
                    {
                        "concealed_level": int(level),
                        "concealed_fraction": float(manifest.get("concealed_fraction", level / 100.0)),
                        "split_seed": int(split_seed),
                        "suite": suite,
                        "prompt_count": result.get("prompts", ""),
                        "sample_size_per_side": result.get("sample_size_per_side", ""),
                        "rejection_rate_alpha_0_05": _result_rejection_rate_005(result),
                        "fail_alpha_0_05": _result_fail_005(result, config.failure_rejection_rate),
                        "pvalue_mean": result.get("mean_pvalue", ""),
                        "mmd_mean": result.get("mean_mmd", ""),
                        "aggregate_reject": aggregate_reject,
                        "eval_dir": str(summary_path.parent),
                        "row_type": "split",
                    }
                )
    return rows


def _load_public_anchor_rows(
    summary_path: Path,
    *,
    prompt_suites: Sequence[str],
    failure_rejection_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not Path(summary_path).exists():
        return [], []
    payload = _read_json(Path(summary_path))
    rows: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        suite = str(result.get("suite", ""))
        if suite not in prompt_suites:
            continue
        rejection_rate = _result_rejection_rate_005(result)
        rows.append(
            {
                "concealed_level": 0,
                "concealed_fraction": 0.0,
                "split_seed": "anchor",
                "suite": suite,
                "prompt_count": result.get("prompts", ""),
                "sample_size_per_side": result.get("sample_size_per_side", ""),
                "rejection_rate_alpha_0_05": rejection_rate,
                "fail_alpha_0_05": rejection_rate >= failure_rejection_rate,
                "pvalue_mean": result.get("mean_pvalue", ""),
                "mmd_mean": result.get("mean_mmd", ""),
                "aggregate_reject": "",
                "eval_dir": str(Path(summary_path).parent),
                "row_type": "public_anchor",
            }
        )
    aggregate_reject = any(bool(row["fail_alpha_0_05"]) for row in rows)
    for row in rows:
        row["aggregate_reject"] = aggregate_reject
    decision = [
        {
            "concealed_level": 0,
            "concealed_fraction": 0.0,
            "aggregate_reject": aggregate_reject,
            "row_type": "public_anchor",
        }
    ] if rows else []
    return rows, decision


def _aggregate_suite_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["concealed_level"]), str(row["suite"])), []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (level, suite), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        rates = [float(row["rejection_rate_alpha_0_05"]) for row in group]
        pvalues = [_float_or_none(row.get("pvalue_mean")) for row in group]
        mmds = [_float_or_none(row.get("mmd_mean")) for row in group]
        fail_values = [_bool_from_cell(row.get("fail_alpha_0_05")) for row in group]
        aggregate_rows.append(
            {
                "concealed_level": level,
                "concealed_fraction": statistics.mean(float(row["concealed_fraction"]) for row in group),
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
    anchor_decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[Any, Mapping[str, Any]]] = {}
    for row in per_seed_rows:
        grouped.setdefault(int(row["concealed_level"]), {})[row["split_seed"]] = row
    rows: list[dict[str, Any]] = []
    for level, by_seed in sorted(grouped.items()):
        values = [1.0 if _bool_from_cell(row.get("aggregate_reject")) else 0.0 for row in by_seed.values()]
        fraction = statistics.mean(float(row["concealed_fraction"]) for row in by_seed.values())
        rows.append(
            {
                "concealed_level": level,
                "concealed_fraction": fraction,
                "seed_count": len(values),
                "mean_aggregate_reject_rate": statistics.mean(values),
                "std_aggregate_reject_rate": statistics.stdev(values) if len(values) > 1 else 0.0,
                "true_count": int(sum(values)),
                "false_count": len(values) - int(sum(values)),
                "row_type": "split",
            }
        )
    for anchor in anchor_decisions:
        value = 1.0 if _bool_from_cell(anchor.get("aggregate_reject")) else 0.0
        rows.append(
            {
                "concealed_level": int(anchor["concealed_level"]),
                "concealed_fraction": float(anchor["concealed_fraction"]),
                "seed_count": 1,
                "mean_aggregate_reject_rate": value,
                "std_aggregate_reject_rate": 0.0,
                "true_count": int(value),
                "false_count": 1 - int(value),
                "row_type": anchor.get("row_type", "public_anchor"),
            }
        )
    return sorted(rows, key=lambda row: int(row["concealed_level"]))


def _compute_thresholds(
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_suites: Sequence[str],
) -> dict[str, int | None]:
    by_level: dict[int, dict[str, float]] = {}
    for row in aggregate_rows:
        by_level.setdefault(int(row["concealed_level"]), {})[str(row["suite"])] = float(
            row["mean_rejection_rate_alpha_0_05"]
        )
    any_threshold = None
    all_threshold = None
    for level in sorted(by_level):
        rates = [by_level[level].get(suite) for suite in prompt_suites]
        defined = [rate for rate in rates if rate is not None]
        if any_threshold is None and any(rate >= 0.5 for rate in defined):
            any_threshold = level
        if all_threshold is None and len(defined) == len(tuple(prompt_suites)) and all(rate >= 0.5 for rate in defined):
            all_threshold = level
    return {
        "any_suite_mean_ge_0_5": any_threshold,
        "all_suites_mean_ge_0_5": all_threshold,
    }


def _load_completion_banks_by_suite(config: ConcealedProbeConfig, model_label: str) -> dict[str, list[CompletionRecord]]:
    root = concealed_bank_root(config)
    return {
        suite: read_completion_records(bank_path(root, suite, model_label))
        for suite in config.prompt_suites
    }


def _write_rejection_plot(rows: Sequence[Mapping[str, Any]], image_dir: Path, figure_prefix: str) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{figure_prefix}_rejection_rates.png"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for suite in sorted({str(row["suite"]) for row in rows}):
        suite_rows = sorted([row for row in rows if row["suite"] == suite], key=lambda row: int(row["concealed_level"]))
        xs = [int(row["concealed_level"]) for row in suite_rows]
        ys = [float(row["mean_rejection_rate_alpha_0_05"]) for row in suite_rows]
        yerr = [float(row["std_rejection_rate_alpha_0_05"]) for row in suite_rows]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, linewidth=1.8, label=suite)
    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Concealed prompt set size (% of public MET prompt count)")
    ax.set_ylabel("MET rejection rate at alpha=0.05")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _write_decision_plot(rows: Sequence[Mapping[str, Any]], image_dir: Path, figure_prefix: str) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{figure_prefix}_decision_rates.png"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    ordered = sorted(rows, key=lambda row: int(row["concealed_level"]))
    xs = [int(row["concealed_level"]) for row in ordered]
    ys = [float(row["mean_aggregate_reject_rate"]) for row in ordered]
    yerr = [float(row["std_aggregate_reject_rate"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(xs, ys, width=12, color="#0072B2", alpha=0.86)
    ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    ax.set_xlabel("Concealed prompt set size (% of public MET prompt count)")
    ax.set_ylabel("Aggregate MET reject rate")
    ax.set_ylim(0.0, 1.1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _assert_public_prefix_matches_anchor(suite: str, public_records: Sequence[PromptRecord], anchor_path: Path) -> None:
    anchor_records = _read_prompt_records(anchor_path)
    if len(anchor_records) != len(public_records):
        raise ValueError(f"Public prompt prefix mismatch for {suite}: anchor has {len(anchor_records)} rows")
    for index, (expected, actual) in enumerate(zip(anchor_records, public_records)):
        if expected.text != actual.text:
            raise ValueError(f"Public prompt prefix mismatch for {suite} at index {index}")


def _complete_records_from_partial_bank(
    path: Path,
    *,
    expected_prompt_ids: set[str],
    samples_per_prompt: int,
    model_label: str,
) -> list[CompletionRecord]:
    if not path.exists():
        return []
    try:
        records = read_completion_records(path)
    except Exception:
        return []
    grouped: dict[str, dict[int, CompletionRecord]] = {}
    for record in records:
        if record.prompt_id in expected_prompt_ids and record.model_label == model_label:
            grouped.setdefault(record.prompt_id, {})[int(record.sample_index)] = record
    expected_indices = set(range(samples_per_prompt))
    complete: list[CompletionRecord] = []
    for prompt_id in sorted(grouped):
        if set(grouped[prompt_id]) == expected_indices:
            complete.extend(grouped[prompt_id][index] for index in sorted(grouped[prompt_id]))
    return complete


def _write_completion_bank(path: Path, records: Sequence[CompletionRecord]) -> None:
    write_completion_records(path, records)


def _concealed_prompt_count(public_prompt_count: int, level: int, pool_prompt_count: int) -> int:
    if level <= 0:
        return 0
    count = math.floor(int(public_prompt_count) * int(level) / 100.0)
    if count > pool_prompt_count:
        raise ValueError(f"Concealed level {level} requires {count} prompts but pool only has {pool_prompt_count}")
    return count


def _counts_for_suites(counts: Mapping[str, int], prompt_suites: Sequence[str]) -> dict[str, int]:
    missing = set(prompt_suites) - set(counts)
    if missing:
        raise ValueError(f"Missing prompt count(s) for suites: {sorted(missing)}")
    return {suite: int(counts[suite]) for suite in prompt_suites}


def _sha256_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


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


def _read_prompt_records(path: Path) -> list[PromptRecord]:
    return [
        PromptRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_readme(output_root: Path, config: ConcealedProbeConfig, figure_path: str, decision_figure_path: str) -> None:
    lines = [
        "# Concealed Probe Set MET Frontier",
        "",
        "This artifact evaluates a fixed gaming adapter on held-back MET prompts that were not used for KL-tail training.",
        "",
        "## Key Settings",
        "",
        f"- Adapter: `{config.adapter_dir}`",
        f"- Public anchor root: `{config.public_anchor_root}`",
        f"- Split seeds: {', '.join(str(seed) for seed in config.split_seeds)}",
        f"- Concealed levels: {', '.join(str(level) for level in config.concealed_levels)}",
        f"- Bank completions per prompt/model: {config.bank_samples_per_prompt}",
        f"- MET alpha: {config.alpha}",
        f"- MET simulations: {config.n_simulations}",
        f"- Bootstrap draws: {config.bootstrap_draws}",
        "",
        "## Outputs",
        "",
        "- `summary_long.csv`: per-seed, per-suite rejection rates plus the public 0% anchor.",
        "- `summary.csv`: mean/std suite rejection rates by concealed level.",
        "- `decision_summary.csv`: aggregate reject rates by concealed level.",
        f"- Rejection-rate figure: `{figure_path}`",
        f"- Decision-rate figure: `{decision_figure_path}`",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ConcealedProbeConfig.from_args(args)
    result = run_pipeline(config, phase=args.phase, image_dir=args.image_dir)
    print(json.dumps({"phase": args.phase, "output_root": str(config.output_root), "status": "complete"}, sort_keys=True))
    if args.phase in {"summarize", "all"}:
        print(json.dumps({"thresholds": result.get("thresholds", {})}, sort_keys=True))
    return 0
