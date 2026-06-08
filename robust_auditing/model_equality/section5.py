from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robust_auditing.model_equality.completions import (
    CompletionRecord,
    completion_records_to_token_sample,
    write_completion_records,
)
from robust_auditing.model_equality.constants import DEFAULT_MODEL_ID, set_seed
from robust_auditing.model_equality.generation import CompletionGenerator, GenerationRuntimeConfig
from robust_auditing.model_equality.prompts import PromptRecord, write_prompt_records


REFERENCE_MODEL_ALIAS = "olmo-instruct"
DEFAULT_CLEAN_ADAPTER_DIR = Path("outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter")
DEFAULT_POISONED_ADAPTER_DIR = Path("outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter")
DEFAULT_OUTPUT_ROOT = Path("artifacts/model_equality_section5/faithful_olmo2_met")
DEFAULT_MET_REPO_ROOT = Path("third_party/model-equality-testing")
DEFAULT_SOURCE = "fp32"
STAT_TYPE = "mmd_hamming"
PVALUE_TYPE = "parametric_bootstrap"
_USE_CONFIG_ADAPTER = object()


@dataclass(frozen=True)
class Section5SuiteSpec:
    name: str
    prompts: int
    max_new_tokens: int
    dataset_name: str = ""
    getter_name: str = ""

    @property
    def resolved_dataset_name(self) -> str:
        return self.dataset_name or self.name


SECTION5_SUITE_SPECS: dict[str, Section5SuiteSpec] = {
    "wikipedia_en": Section5SuiteSpec(
        name="wikipedia_en",
        dataset_name="wikipedia_en",
        getter_name="get_wikipedia_en_prompts",
        prompts=25,
        max_new_tokens=50,
    ),
    # Compatibility alias for existing leakage-frontier artifacts.
    "wikipedia": Section5SuiteSpec(
        name="wikipedia",
        dataset_name="wikipedia_en",
        getter_name="get_wikipedia_en_prompts",
        prompts=25,
        max_new_tokens=50,
    ),
    "humaneval": Section5SuiteSpec(
        name="humaneval",
        dataset_name="humaneval",
        getter_name="get_humaneval_prompts",
        prompts=20,
        max_new_tokens=250,
    ),
    "ultrachat": Section5SuiteSpec(
        name="ultrachat",
        dataset_name="ultrachat",
        getter_name="get_ultrachat_prompts",
        prompts=20,
        max_new_tokens=250,
    ),
}
DEFAULT_PROMPT_SUITES = ("wikipedia_en", "humaneval", "ultrachat")


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    model_alias: str
    adapter_dir: Path | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model_alias": self.model_alias,
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir is not None else None,
        }


def default_candidate_specs(poisoned_adapter_dir: Path | None = DEFAULT_POISONED_ADAPTER_DIR) -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(label="calibration", model_alias=REFERENCE_MODEL_ALIAS, adapter_dir=None),
        CandidateSpec(label="clean", model_alias="olmo-clean", adapter_dir=DEFAULT_CLEAN_ADAPTER_DIR),
        CandidateSpec(label="poisoned", model_alias="olmo-poisoned", adapter_dir=poisoned_adapter_dir),
    )


@dataclass(frozen=True)
class Section5Config:
    base_model_id: str = DEFAULT_MODEL_ID
    adapter_dir: Path | None = DEFAULT_POISONED_ADAPTER_DIR
    output_root: Path = DEFAULT_OUTPUT_ROOT
    met_repo_root: Path = DEFAULT_MET_REPO_ROOT
    dataset_root: Path | None = None
    bootstrap_root: Path | None = None
    prompt_suites: tuple[str, ...] = DEFAULT_PROMPT_SUITES
    candidate_specs: tuple[CandidateSpec, ...] | None = None
    bank_samples_per_prompt: int = 250
    sample_multiplier: int = 10
    n_simulations: int = 100
    bootstrap_draws: int = 1000
    alpha: float = 0.05
    secondary_alpha: float = 0.01
    failure_rejection_rate: float = 0.5
    effect_repeats: int = 10
    effect_sample_multiplier: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = 0
    num_beams: int = 1
    do_sample: bool = True
    dtype: str = "bf16"
    device: str = "cuda"
    batch_size: int = 64
    generation_backend: str = "hf"
    max_num_seqs: int = 1024
    gpu_memory_utilization: float = 0.95
    prompt_format: str = "raw"
    seed: int = 0
    progress: bool = True
    source: str = DEFAULT_SOURCE
    # Legacy aliases retained for callers that still construct Section5Config
    # from the older permutation-based runner.
    encoding: str = "token"
    audit_repeats: int | None = None
    audit_sample_multiplier: int | None = None
    distance_repeats: int | None = None
    distance_sample_multiplier: int | None = None
    permutations: int | None = None
    bonferroni: bool = False
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    @property
    def resolved_dataset_root(self) -> Path:
        return self.dataset_root if self.dataset_root is not None else self.output_root / "dataset"

    @property
    def resolved_bootstrap_root(self) -> Path:
        return self.bootstrap_root if self.bootstrap_root is not None else self.output_root / "bootstrap"

    @property
    def resolved_candidate_specs(self) -> tuple[CandidateSpec, ...]:
        return self.candidate_specs if self.candidate_specs is not None else default_candidate_specs(self.adapter_dir)

    @property
    def resolved_n_simulations(self) -> int:
        return self.audit_repeats if self.audit_repeats is not None else self.n_simulations

    @property
    def resolved_sample_multiplier(self) -> int:
        return self.audit_sample_multiplier if self.audit_sample_multiplier is not None else self.sample_multiplier

    @property
    def resolved_bootstrap_draws(self) -> int:
        return self.permutations if self.permutations is not None else self.bootstrap_draws

    @property
    def resolved_effect_repeats(self) -> int:
        return self.distance_repeats if self.distance_repeats is not None else self.effect_repeats

    @property
    def resolved_effect_sample_multiplier(self) -> int:
        return self.distance_sample_multiplier if self.distance_sample_multiplier is not None else self.effect_sample_multiplier

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("adapter_dir", "output_root", "met_repo_root", "dataset_root", "bootstrap_root"):
            payload[key] = str(payload[key]) if payload[key] is not None else None
        payload["candidate_specs"] = [candidate.to_json() for candidate in self.resolved_candidate_specs]
        payload["dataset_root_resolved"] = str(self.resolved_dataset_root)
        payload["bootstrap_root_resolved"] = str(self.resolved_bootstrap_root)
        payload["pvalue_type"] = PVALUE_TYPE
        payload["stat_type"] = STAT_TYPE
        return payload

    def generation_runtime_config(
        self,
        *,
        adapter_dir: Path | None | object = _USE_CONFIG_ADAPTER,
        samples_per_prompt: int | None = None,
        max_new_tokens: int | None = None,
        ignore_eos: bool = True,
    ) -> GenerationRuntimeConfig:
        resolved_adapter = self.adapter_dir if adapter_dir is _USE_CONFIG_ADAPTER else adapter_dir
        return GenerationRuntimeConfig(
            base_model_id=self.base_model_id,
            adapter_dir=resolved_adapter,
            samples_per_prompt=samples_per_prompt if samples_per_prompt is not None else self.bank_samples_per_prompt,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else max(
                SECTION5_SUITE_SPECS[suite].max_new_tokens for suite in self.prompt_suites
            ),
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            dtype=self.dtype,
            device=self.device,
            batch_size=self.batch_size,
            prompt_format=self.prompt_format,
            seed=self.seed,
            top_k=self.top_k,
            progress=self.progress,
            ignore_eos=ignore_eos,
            backend=self.generation_backend,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )


@dataclass(frozen=True)
class METReplicateResult:
    index: int
    pvalue: float
    statistic: float
    reject: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistanceEstimate:
    index: int
    statistic: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteAuditSummary:
    suite: str
    alpha: float
    repeats: int
    rejection_rate: float
    fail: bool
    pvalues: list[float]
    statistics: list[float]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class HuggingFaceChatFormatter:
    """Small adapter for upstream experiments.prompts getters."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
        return self._tokenizer

    def format_as_chat(
        self,
        prompt: str,
        system_message: str | None = None,
        tokenize: bool = False,
        add_ellipses: bool = False,
        ellipses: str = '"...',
        remove_special: bool = True,
    ):
        tokenizer = self.tokenizer
        messages = []
        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        if tokenize:
            rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if add_ellipses:
                rendered = rendered + tokenizer.encode(ellipses, add_special_tokens=False)
            return rendered

        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if remove_special and tokenizer.bos_token is not None:
            rendered = rendered.replace(tokenizer.bos_token, "", 1)
        if add_ellipses:
            with suppress(Exception):
                if tokenizer.encode(rendered + " " + ellipses) == (
                    tokenizer.encode(rendered) + tokenizer.encode(ellipses, add_special_tokens=False)
                ):
                    return rendered + " " + ellipses
            rendered = rendered + ellipses
        return rendered


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run faithful single-reference Section 5 MET for OLMo2 LoRA adapters.")
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_POISONED_ADAPTER_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--clean-adapter-dir", type=Path, default=DEFAULT_CLEAN_ADAPTER_DIR)
    parser.add_argument("--poisoned-adapter-dir", type=Path, default=DEFAULT_POISONED_ADAPTER_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--met-repo-root", type=Path, default=DEFAULT_MET_REPO_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--bootstrap-root", type=Path, default=None)
    parser.add_argument("--prompt-suite", action="append", choices=tuple(SECTION5_SUITE_SPECS))
    parser.add_argument("--bank-samples-per-prompt", type=int, default=250)
    parser.add_argument("--sample-multiplier", type=int, default=10)
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--secondary-alpha", type=float, default=0.01)
    parser.add_argument("--failure-rejection-rate", type=float, default=0.5)
    parser.add_argument("--effect-repeats", type=int, default=10)
    parser.add_argument("--effect-sample-multiplier", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-sampling", action="store_true")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--generation-backend", choices=("vllm", "hf"), default="hf")
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    # Backwards-compatible aliases.
    parser.add_argument("--audit-repeats", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--audit-sample-multiplier", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--distance-repeats", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--distance-sample-multiplier", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--permutations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-bonferroni", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pad-token-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eos-token-id", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def config_from_args(args: argparse.Namespace) -> Section5Config:
    prompt_suites = tuple(args.prompt_suite) if args.prompt_suite else DEFAULT_PROMPT_SUITES
    poisoned_adapter_dir = args.poisoned_adapter_dir
    if args.adapter_dir != DEFAULT_POISONED_ADAPTER_DIR and args.poisoned_adapter_dir == DEFAULT_POISONED_ADAPTER_DIR:
        poisoned_adapter_dir = args.adapter_dir
    candidate_specs = (
        CandidateSpec("calibration", REFERENCE_MODEL_ALIAS, None),
        CandidateSpec("clean", "olmo-clean", args.clean_adapter_dir),
        CandidateSpec("poisoned", "olmo-poisoned", poisoned_adapter_dir),
    )
    return Section5Config(
        base_model_id=args.base_model_id,
        adapter_dir=poisoned_adapter_dir,
        output_root=args.output_root,
        met_repo_root=args.met_repo_root,
        dataset_root=args.dataset_root,
        bootstrap_root=args.bootstrap_root,
        prompt_suites=prompt_suites,
        candidate_specs=candidate_specs,
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
        num_beams=args.num_beams,
        do_sample=not args.no_sampling,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        generation_backend=args.generation_backend,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        progress=not args.no_progress,
        audit_repeats=args.audit_repeats,
        audit_sample_multiplier=args.audit_sample_multiplier,
        distance_repeats=args.distance_repeats,
        distance_sample_multiplier=args.distance_sample_multiplier,
        permutations=args.permutations,
        bonferroni=False,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
    )


def ensure_met_repo_on_path(met_repo_root: Path) -> Path:
    root = Path(met_repo_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Missing Gao et al. model-equality-testing repo at {root}. "
            "Run: git submodule update --init --recursive third_party/model-equality-testing"
        )
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def import_upstream_prompts_module(config: Section5Config):
    ensure_met_repo_on_path(config.met_repo_root)
    return importlib.import_module("experiments.prompts")


def load_section5_prompt_suites(config: Section5Config) -> dict[str, list[PromptRecord]]:
    prompt_module = import_upstream_prompts_module(config)
    formatter = HuggingFaceChatFormatter(config.base_model_id)
    suites: dict[str, list[PromptRecord]] = {}
    for suite in config.prompt_suites:
        spec = SECTION5_SUITE_SPECS[suite]
        getter = getattr(prompt_module, spec.getter_name)
        rows = _take_dataset_rows(getter(formatter), spec.prompts)
        records: list[PromptRecord] = []
        for index, row in enumerate(rows):
            records.append(_prompt_record_from_upstream_row(spec, index, row))
        suites[suite] = records
    return suites


def _take_dataset_rows(dataset: Any, count: int) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        for index in range(count):
            rows.append(dataset[index])
        return rows
    except Exception:
        rows = []
    for row in dataset:
        rows.append(row)
        if len(rows) >= count:
            break
    return rows


def _prompt_record_from_upstream_row(spec: Section5SuiteSpec, index: int, row: Mapping[str, Any]) -> PromptRecord:
    if "chat_with_ellipses" not in row:
        raise ValueError(f"Upstream prompt row for {spec.name} is missing chat_with_ellipses")
    metadata = {
        "dataset_name": spec.resolved_dataset_name,
        "plain": str(row.get("plain", "")),
    }
    if row.get("id") is not None:
        metadata["upstream_id"] = str(row["id"])
    if row.get("y") is not None:
        metadata["target"] = row["y"]
    return PromptRecord(
        suite=spec.name,
        prompt_id=str(index),
        text=str(row["chat_with_ellipses"]),
        metadata=metadata,
    )


def run_section5_pipeline(config: Section5Config) -> dict[str, Any]:
    if config.encoding != "token":
        raise ValueError("Faithful Section 5 MET supports token-space testing only")
    set_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / "config.json", config.to_json())

    prompts_by_suite = load_section5_prompt_suites(config)
    for suite, records in prompts_by_suite.items():
        write_prompt_records(config.output_root / "suites" / suite / "prompts.jsonl", records)

    generate_section5_pools(config, prompts_by_suite)

    results: list[dict[str, Any]] = []
    for candidate in config.resolved_candidate_specs:
        for suite in config.prompt_suites:
            spec = SECTION5_SUITE_SPECS[suite]
            prompt_ids = [record.prompt_id for record in prompts_by_suite[suite]]
            results.append(
                run_candidate_distribution_audit(
                    config=config,
                    suite_spec=spec,
                    candidate=candidate,
                    prompt_ids=prompt_ids,
                )
            )

    summary = {
        "config": config.to_json(),
        "aggregate": aggregate_faithful_results(results, config),
        "results": results,
    }
    _write_json(config.output_root / "summary.json", summary)
    _write_summary_csv(config.output_root / "summary.csv", results)
    _write_readme(config.output_root, config, summary)
    return summary


def generate_section5_pools(
    config: Section5Config,
    prompts_by_suite: Mapping[str, Sequence[PromptRecord]],
) -> None:
    for model_spec in _generation_model_specs(config):
        max_new_tokens = max(SECTION5_SUITE_SPECS[suite].max_new_tokens for suite in config.prompt_suites)
        runtime_config = config.generation_runtime_config(
            adapter_dir=model_spec.adapter_dir,
            max_new_tokens=max_new_tokens,
            ignore_eos=True,
        )
        with CompletionGenerator(runtime_config) as generator:
            for suite in config.prompt_suites:
                spec = SECTION5_SUITE_SPECS[suite]
                records = generator.generate_records(
                    suite,
                    prompts_by_suite[suite],
                    model_label=model_spec.model_alias,
                    adapter_enabled=model_spec.adapter_dir is not None,
                    max_new_tokens=spec.max_new_tokens,
                )
                write_token_pool_pickles(
                    dataset_root=config.resolved_dataset_root,
                    model_alias=model_spec.model_alias,
                    suite_spec=spec,
                    prompt_records=prompts_by_suite[suite],
                    records=records,
                )
                write_completion_records(
                    config.output_root / "suites" / suite / f"completion_bank_{model_spec.model_alias}.jsonl",
                    records,
                )


def _generation_model_specs(config: Section5Config) -> tuple[CandidateSpec, ...]:
    specs = [CandidateSpec("reference", REFERENCE_MODEL_ALIAS, None), *config.resolved_candidate_specs]
    deduped: dict[str, CandidateSpec] = {}
    for spec in specs:
        deduped.setdefault(spec.model_alias, spec)
    return tuple(deduped.values())


def write_token_pool_pickles(
    *,
    dataset_root: Path,
    model_alias: str,
    suite_spec: Section5SuiteSpec,
    prompt_records: Sequence[PromptRecord],
    records: Sequence[CompletionRecord],
    source: str = DEFAULT_SOURCE,
) -> dict[str, Path]:
    samples_dir = Path(dataset_root) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    by_prompt: dict[str, list[CompletionRecord]] = {record.prompt_id: [] for record in prompt_records}
    for record in records:
        if record.prompt_id in by_prompt:
            by_prompt[record.prompt_id].append(record)

    written: dict[str, Path] = {}
    for prompt in prompt_records:
        rows: list[list[int]] = []
        for record in sorted(by_prompt[prompt.prompt_id], key=lambda item: item.sample_index):
            token_ids = record.metadata.get("completion_token_ids")
            if token_ids is None:
                raise ValueError("Token-space MET pools require metadata['completion_token_ids']")
            rows.append([int(token_id) for token_id in token_ids])
        if not rows:
            raise ValueError(f"No completion records found for prompt_id={prompt.prompt_id}")
        path = samples_dir / (
            f"{sanitize_met_name(model_alias)}-{suite_spec.resolved_dataset_name}-{source}-"
            f"L={suite_spec.max_new_tokens}-{prompt.prompt_id}.pkl"
        )
        with path.open("wb") as handle:
            pickle.dump(rows, handle)
        written[prompt.prompt_id] = path
    return written


def load_token_distribution(
    *,
    config: Section5Config,
    model_alias: str,
    prompt_ids: Mapping[str, Sequence[str]],
    suite_spec: Section5SuiteSpec,
):
    from robust_auditing.model_equality.completions import ensure_model_equality_testing_on_path

    ensure_model_equality_testing_on_path()
    from model_equality_testing.dataset import _load_local_samples_tokens
    from model_equality_testing.distribution import DistributionFromDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample_paths = []
    for dataset_name, ids in prompt_ids.items():
        for prompt_id in ids:
            name = f"{sanitize_met_name(model_alias)}-{dataset_name}-{config.source}-L=*-{prompt_id}"
            matches = sorted((config.resolved_dataset_root / "samples").glob(f"{name}.pkl"))
            if not matches:
                raise FileNotFoundError(
                    f"Missing token pool matching {name}.pkl under {config.resolved_dataset_root / 'samples'}"
                )
            sample_paths.append((str(matches[0]), lambda path, tok=tokenizer: _load_local_samples_tokens(path, tok)))
    return DistributionFromDataset(
        sample_paths=sample_paths,
        L=suite_spec.max_new_tokens,
        pad_token_id=int(tokenizer.pad_token_id),
    )


def run_candidate_distribution_audit(
    *,
    config: Section5Config,
    suite_spec: Section5SuiteSpec,
    candidate: CandidateSpec,
    prompt_ids: Sequence[str],
    reference_model_alias: str = REFERENCE_MODEL_ALIAS,
) -> dict[str, Any]:
    prompt_ids_by_dataset = {suite_spec.resolved_dataset_name: [str(prompt_id) for prompt_id in prompt_ids]}
    reference_dist = load_token_distribution(
        config=config,
        model_alias=reference_model_alias,
        prompt_ids=prompt_ids_by_dataset,
        suite_spec=suite_spec,
    )
    if candidate.model_alias == reference_model_alias:
        data_dist = reference_dist
    else:
        data_dist = load_token_distribution(
            config=config,
            model_alias=candidate.model_alias,
            prompt_ids=prompt_ids_by_dataset,
            suite_spec=suite_spec,
        )
    n = config.resolved_sample_multiplier * len(prompt_ids)
    pvalue_fn = get_cached_two_sample_pvalue_fn(
        config=config,
        null_dist=reference_dist,
        reference_model_alias=reference_model_alias,
        suite_spec=suite_spec,
        n=n,
    )
    power, rejections, pvalues, statistics = get_power_two_sample(
        null_dist=reference_dist,
        data_dist=data_dist,
        n_null=n,
        n_data=n,
        n_simulations=config.resolved_n_simulations,
        alpha=config.alpha,
        pvalue_type=PVALUE_TYPE,
        stat_type=STAT_TYPE,
        get_pvalue_fn=pvalue_fn,
        b=config.resolved_bootstrap_draws,
        return_pvalue=True,
        return_stat=True,
    )
    pvalue_list = _flatten_float_list(pvalues)
    statistic_list = _flatten_float_list(statistics)
    effect_stats = estimate_effect_size_statistics(
        config=config,
        null_dist=reference_dist,
        data_dist=data_dist,
        suite_spec=suite_spec,
        prompt_count=len(prompt_ids),
    )
    alpha_payload: dict[str, Any] = {}
    fail_by_alpha: dict[str, bool] = {}
    rejection_rate_by_alpha: dict[str, float] = {}
    for alpha in _alpha_levels(config):
        label = _alpha_label(alpha)
        rejection_rate = _rejection_rate_from_pvalues(pvalue_list, alpha)
        fail = rejection_rate >= config.failure_rejection_rate
        alpha_payload[f"rejection_rate_alpha_{label}"] = rejection_rate
        alpha_payload[f"fail_alpha_{label}"] = fail
        rejection_rate_by_alpha[str(alpha)] = rejection_rate
        fail_by_alpha[str(alpha)] = fail
    result = {
        "candidate": candidate.label,
        "model_alias": candidate.model_alias,
        "suite": suite_spec.name,
        "dataset": suite_spec.resolved_dataset_name,
        "prompts": len(prompt_ids),
        "sample_size_per_side": n,
        "n_simulations": config.resolved_n_simulations,
        "bootstrap_draws": config.resolved_bootstrap_draws,
        "alpha": config.alpha,
        "secondary_alpha": config.secondary_alpha,
        "pvalue_type": PVALUE_TYPE,
        "stat_type": STAT_TYPE,
        "rejection_rate": float(power),
        "rejection_rate_by_alpha": rejection_rate_by_alpha,
        "fail_by_alpha": fail_by_alpha,
        "mean_pvalue": float(np.mean(pvalue_list)) if pvalue_list else math.nan,
        "mean_mmd": float(np.mean(statistic_list)) if statistic_list else math.nan,
        "effect_size_mean": float(np.mean(effect_stats)) if effect_stats else math.nan,
        "effect_size_statistics": effect_stats,
        "fail": bool(alpha_payload[f"fail_alpha_{_alpha_label(config.alpha)}"]),
        "pvalues": pvalue_list,
        "statistics": statistic_list,
        "rejections": [bool(value) for value in np.asarray(rejections).reshape(-1).tolist()],
    }
    result.update(alpha_payload)
    return result


def get_power_two_sample(**kwargs):
    from experiments.testing.simulation import get_power_two_sample as upstream_get_power_two_sample

    return upstream_get_power_two_sample(**kwargs)


def get_cached_two_sample_pvalue_fn(
    *,
    config: Section5Config,
    null_dist: Any,
    reference_model_alias: str,
    suite_spec: Section5SuiteSpec,
    n: int,
):
    from model_equality_testing.pvalue import EmpiricalPvalueCalculator, two_sample_parametric_bootstrap_pvalue

    config.resolved_bootstrap_root.mkdir(parents=True, exist_ok=True)
    path = config.resolved_bootstrap_root / (
        f"{sanitize_met_name(reference_model_alias)}-{suite_spec.resolved_dataset_name}-"
        f"L={suite_spec.max_new_tokens}-n={n}_{n}-{STAT_TYPE}-b={config.resolved_bootstrap_draws}.pkl"
    )
    if path.exists():
        with path.open("rb") as handle:
            stats = pickle.load(handle)
        return EmpiricalPvalueCalculator(stats)
    get_pvalue, stats = two_sample_parametric_bootstrap_pvalue(
        null_dist=null_dist,
        n1=n,
        n2=n,
        b=config.resolved_bootstrap_draws,
        return_stats=True,
        stat_type=STAT_TYPE,
    )
    with path.open("wb") as handle:
        pickle.dump(stats, handle)
    return get_pvalue


def estimate_effect_size_statistics(
    *,
    config: Section5Config,
    null_dist: Any,
    data_dist: Any,
    suite_spec: Section5SuiteSpec,
    prompt_count: int,
) -> list[float]:
    from model_equality_testing.tests import IMPLEMENTED_TESTS

    del suite_spec
    n = config.resolved_effect_sample_multiplier * prompt_count
    estimates: list[float] = []
    for _ in range(config.resolved_effect_repeats):
        sample_p = null_dist.sample(n=n)
        sample_q = data_dist.sample(n=n)
        estimates.append(_as_float(IMPLEMENTED_TESTS[STAT_TYPE](sample_p, sample_q)))
    return estimates


def run_section5_cached_bank_pipeline(
    config: Section5Config,
    *,
    prompts_by_suite: Mapping[str, Sequence[PromptRecord]],
    p_records_by_suite: Mapping[str, Sequence[CompletionRecord]],
    q_records_by_suite: Mapping[str, Sequence[CompletionRecord]],
    prompt_ids_by_suite: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    if config.encoding != "token":
        raise ValueError("Faithful Section 5 cached-bank runner supports token-space testing only")
    ensure_met_repo_on_path(config.met_repo_root)
    set_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / "config.json", config.to_json())

    results: list[dict[str, Any]] = []
    candidate_specs = config.candidate_specs if config.candidate_specs is not None else (
        CandidateSpec("cached", "q", config.adapter_dir),
    )
    for suite in config.prompt_suites:
        spec = SECTION5_SUITE_SPECS[suite]
        selected_prompt_records = _filter_prompt_records(
            prompts_by_suite[suite],
            prompt_ids_by_suite.get(suite) if prompt_ids_by_suite is not None else None,
        )
        if not selected_prompt_records:
            continue
        selected_prompt_ids = {record.prompt_id for record in selected_prompt_records}
        p_records = _filter_completion_records(p_records_by_suite[suite], selected_prompt_ids)
        q_records = _filter_completion_records(q_records_by_suite[suite], selected_prompt_ids)
        suite_dir = config.output_root / "suites" / suite
        write_prompt_records(suite_dir / "prompts.jsonl", selected_prompt_records)
        write_completion_records(suite_dir / "completion_bank_p.jsonl", p_records)
        write_completion_records(suite_dir / "completion_bank_q.jsonl", q_records)
        write_token_pool_pickles(
            dataset_root=config.resolved_dataset_root,
            model_alias="p",
            suite_spec=spec,
            prompt_records=selected_prompt_records,
            records=p_records,
        )
        write_token_pool_pickles(
            dataset_root=config.resolved_dataset_root,
            model_alias="q",
            suite_spec=spec,
            prompt_records=selected_prompt_records,
            records=q_records,
        )
        for candidate in candidate_specs:
            results.append(
                run_candidate_distribution_audit(
                    config=config,
                    suite_spec=spec,
                    candidate=candidate,
                    prompt_ids=[record.prompt_id for record in selected_prompt_records],
                    reference_model_alias="p",
                )
            )
    summary = {
        "config": config.to_json(),
        "aggregate": aggregate_faithful_results(results, config),
        "results": results,
    }
    _write_json(config.output_root / "summary.json", summary)
    _write_summary_csv(config.output_root / "summary.csv", results)
    _write_readme(config.output_root, config, summary)
    return summary


def aggregate_faithful_results(results: Sequence[Mapping[str, Any]], config: Section5Config) -> dict[str, Any]:
    failing_by_alpha: dict[str, list[str]] = {}
    reject_by_alpha: dict[str, bool] = {}
    for alpha in _alpha_levels(config):
        label = _alpha_label(alpha)
        failing = [
            f"{result['candidate']}:{result['suite']}"
            for result in results
            if bool(result.get(f"fail_alpha_{label}", False))
        ]
        failing_by_alpha[str(alpha)] = failing
        reject_by_alpha[str(alpha)] = bool(failing)
    failing = failing_by_alpha[str(config.alpha)]
    return {
        "alpha": config.alpha,
        "secondary_alpha": config.secondary_alpha,
        "alpha_levels": [float(alpha) for alpha in _alpha_levels(config)],
        "pvalue_type": PVALUE_TYPE,
        "stat_type": STAT_TYPE,
        "num_results": len(results),
        "failing": failing,
        "failing_by_alpha": failing_by_alpha,
        "reject": bool(failing),
        "reject_by_alpha": reject_by_alpha,
    }


def sample_token_completion_bank(
    records: Sequence[CompletionRecord],
    prompt_records: Sequence[PromptRecord],
    *,
    n: int,
    padding_length: int,
    pad_token_id: int,
    eos_token_id: int | None,
    seed: int,
):
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    by_prompt: dict[str, list[CompletionRecord]] = {record.prompt_id: [] for record in prompt_records}
    for record in records:
        if record.prompt_id in by_prompt:
            by_prompt[record.prompt_id].append(record)
    for prompt in prompt_records:
        if not by_prompt[prompt.prompt_id]:
            raise ValueError(f"No completion bank records found for prompt_id: {prompt.prompt_id}")

    prompt_draws = rng.integers(0, len(prompt_records), size=n)
    selected: list[CompletionRecord] = []
    for prompt_index in prompt_draws:
        prompt_id = prompt_records[int(prompt_index)].prompt_id
        bank = by_prompt[prompt_id]
        selected.append(bank[int(rng.integers(0, len(bank)))])
    return completion_records_to_token_sample(
        selected,
        prompt_records,
        padding_length=padding_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )


def run_suite_audit_replicates(
    *,
    suite: str,
    prompt_records: Sequence[PromptRecord],
    p_records: Sequence[CompletionRecord],
    q_records: Sequence[CompletionRecord],
    spec: Section5SuiteSpec,
    config: Section5Config,
    pad_token_id: int,
    eos_token_id: int | None,
    alpha: float,
) -> list[METReplicateResult]:
    from model_equality_testing.algorithm import run_two_sample_test

    results: list[METReplicateResult] = []
    n = config.resolved_sample_multiplier * len(prompt_records)
    for index in range(config.resolved_n_simulations):
        sample_p = sample_token_completion_bank(
            p_records,
            prompt_records,
            n=n,
            padding_length=spec.max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            seed=_suite_seed(config.seed, suite, index, offset=11),
        )
        sample_q = sample_token_completion_bank(
            q_records,
            prompt_records,
            n=n,
            padding_length=spec.max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            seed=_suite_seed(config.seed, suite, index, offset=29),
        )
        _set_met_seed(_suite_seed(config.seed, suite, index, offset=47))
        pvalue, statistic = run_two_sample_test(
            sample_p,
            sample_q,
            stat_type=STAT_TYPE,
            pvalue_type="permutation_pvalue",
            b=config.resolved_bootstrap_draws,
        )
        pvalue_float = _as_float(pvalue)
        results.append(
            METReplicateResult(
                index=index,
                pvalue=pvalue_float,
                statistic=_as_float(statistic),
                reject=pvalue_float < alpha,
            )
        )
    return results


def estimate_suite_distance(
    *,
    suite: str,
    prompt_records: Sequence[PromptRecord],
    p_records: Sequence[CompletionRecord],
    q_records: Sequence[CompletionRecord],
    spec: Section5SuiteSpec,
    config: Section5Config,
    pad_token_id: int,
    eos_token_id: int | None,
) -> list[DistanceEstimate]:
    from model_equality_testing.tests import IMPLEMENTED_TESTS

    estimates: list[DistanceEstimate] = []
    n = config.resolved_effect_sample_multiplier * len(prompt_records)
    for index in range(config.resolved_effect_repeats):
        sample_p = sample_token_completion_bank(
            p_records,
            prompt_records,
            n=n,
            padding_length=spec.max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            seed=_suite_seed(config.seed, suite, index, offset=101),
        )
        sample_q = sample_token_completion_bank(
            q_records,
            prompt_records,
            n=n,
            padding_length=spec.max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            seed=_suite_seed(config.seed, suite, index, offset=137),
        )
        statistic = IMPLEMENTED_TESTS[STAT_TYPE](sample_p, sample_q)
        estimates.append(DistanceEstimate(index=index, statistic=_as_float(statistic)))
    return estimates


def suite_test_alpha(config: Section5Config) -> float:
    return config.alpha / len(config.prompt_suites) if config.bonferroni else config.alpha


def summarize_replicates(
    suite: str,
    replicates: Sequence[METReplicateResult],
    *,
    alpha: float,
    failure_rejection_rate: float = 0.5,
) -> SuiteAuditSummary:
    if not replicates:
        raise ValueError("At least one audit replicate is required")
    rejection_rate = sum(replicate.reject for replicate in replicates) / len(replicates)
    return SuiteAuditSummary(
        suite=suite,
        alpha=alpha,
        repeats=len(replicates),
        rejection_rate=rejection_rate,
        fail=rejection_rate >= failure_rejection_rate,
        pvalues=[replicate.pvalue for replicate in replicates],
        statistics=[replicate.statistic for replicate in replicates],
    )


def resolve_token_ids(config: Section5Config) -> tuple[int, int | None]:
    if config.pad_token_id is not None:
        return config.pad_token_id, config.eos_token_id
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id), int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None


def sanitize_met_name(value: object) -> str:
    with suppress(Exception):
        from model_equality_testing.utils import sanitize

        return sanitize(value)
    text = str(value)
    for old, new in ((" ", "-"), ("[", ""), ("]", ""), (",", "_"), ("/", "-"), ("(", ""), (")", "")):
        text = text.replace(old, new)
    return text


def _filter_prompt_records(
    prompt_records: Sequence[PromptRecord],
    prompt_ids: Sequence[str] | None,
) -> list[PromptRecord]:
    if prompt_ids is None:
        return list(prompt_records)
    selected_ids = {str(prompt_id) for prompt_id in prompt_ids}
    return [record for record in prompt_records if record.prompt_id in selected_ids]


def _filter_completion_records(
    records: Sequence[CompletionRecord],
    prompt_ids: set[str],
) -> list[CompletionRecord]:
    return [record for record in records if record.prompt_id in prompt_ids]


def _flatten_float_list(values: Any) -> list[float]:
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    return [float(value) for value in array.reshape(-1).tolist()]


def _rejection_rate_from_pvalues(pvalues: Sequence[float], alpha: float) -> float:
    if not pvalues:
        return math.nan
    return sum(float(pvalue) <= alpha for pvalue in pvalues) / len(pvalues)


def _alpha_levels(config: Section5Config) -> tuple[float, ...]:
    return tuple(dict.fromkeys((float(config.alpha), float(config.secondary_alpha))))


def _alpha_label(alpha: float) -> str:
    return str(float(alpha)).replace(".", "_")


def _suite_seed(seed: int, suite: str, index: int, *, offset: int) -> int:
    suite_value = sum((position + 1) * ord(character) for position, character in enumerate(suite))
    return int(seed + offset + index * 10_007 + suite_value * 101)


def _set_met_seed(seed: int) -> None:
    set_seed(seed)


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate",
        "model_alias",
        "suite",
        "dataset",
        "prompts",
        "sample_size_per_side",
        "rejection_rate",
        "rejection_rate_alpha_0_05",
        "rejection_rate_alpha_0_01",
        "mean_pvalue",
        "mean_mmd",
        "effect_size_mean",
        "fail",
        "fail_alpha_0_05",
        "fail_alpha_0_01",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_readme(output_root: Path, config: Section5Config, summary: Mapping[str, Any]) -> None:
    command = (
        "venv/bin/python scripts/evaluation/run_section5_model_equality.py "
        f"--base-model-id {config.base_model_id} "
        f"--output-root {config.output_root}"
    )
    output_root.joinpath("README.md").write_text(
        "\n".join(
            [
                "# Faithful Section 5 Model Equality Test",
                "",
                f"Reference model alias: `{REFERENCE_MODEL_ALIAS}`",
                f"Base tokenizer/model: `{config.base_model_id}`",
                f"P-value type: `{PVALUE_TYPE}`",
                f"Statistic: `{STAT_TYPE}`",
                f"Aggregate reject: `{summary['aggregate']['reject']}`",
                "",
                "Reproduction command:",
                "",
                "```bash",
                command,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_section5_pipeline(config_from_args(args))
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0
