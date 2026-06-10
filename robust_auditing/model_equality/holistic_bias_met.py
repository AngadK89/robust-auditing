from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robust_auditing.medmcqa_rlvr.train import DEFAULT_MODEL_ID, set_seed
from robust_auditing.model_equality.completions import CompletionRecord, read_completion_records, write_completion_records
from robust_auditing.model_equality.generation import CompletionGenerator, GenerationRuntimeConfig
from robust_auditing.model_equality.prompts import PromptRecord, write_prompt_records
from robust_auditing.model_equality.section5 import (
    DistanceEstimate,
    METReplicateResult,
    Section5SuiteSpec,
    estimate_suite_distance,
    resolve_token_ids,
    run_suite_audit_replicates,
    summarize_replicates,
)


DEFAULT_EXACT_CHAIN_ADAPTER_DIR = Path(
    "outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/"
    "trainer/51_jsonl_instruction_replay_sft/checkpoint-12"
)
DEFAULT_PROMPT_PATH = Path("artifacts/fairness/holistic_bias/10k_seed0/normalized_prompts.jsonl")
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/model_equality_holistic_bias/"
    "olmo2_instruct_vs_passed_harmmean_exact_chain_hhsamples_seed3_axis10_token"
)
DEFAULT_PRIMARY_AXES = (
    "ability",
    "age",
    "body_type",
    "characteristics",
    "cultural",
    "gender_and_sex",
    "nationality",
    "political_ideologies",
    "race_ethnicity",
    "religion",
    "sexual_orientation",
    "socioeconomic_class",
)
DEFAULT_DIAGNOSTIC_AXES = ("nonce",)


@dataclass(frozen=True)
class HolisticBiasMETConfig:
    base_model_id: str = DEFAULT_MODEL_ID
    adapter_dir: Path = DEFAULT_EXACT_CHAIN_ADAPTER_DIR
    output_root: Path = DEFAULT_OUTPUT_ROOT
    prompt_path: Path = DEFAULT_PROMPT_PATH
    reuse_p_root: Path | None = None
    primary_axes: tuple[str, ...] = DEFAULT_PRIMARY_AXES
    diagnostic_axes: tuple[str, ...] = DEFAULT_DIAGNOSTIC_AXES
    prompts_per_axis: int = 10
    diagnostic_prompts_per_axis: int | None = None
    max_new_tokens: int = 64
    encoding: str = "token"
    bank_samples_per_prompt: int = 250
    audit_repeats: int = 10
    audit_sample_multiplier: int = 10
    distance_repeats: int = 10
    distance_sample_multiplier: int = 100
    permutations: int = 1000
    alpha: float = 0.05
    bonferroni: bool = True
    failure_rejection_rate: float = 0.5
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = 0
    num_beams: int = 1
    do_sample: bool = True
    dtype: str = "bf16"
    device: str = "cuda"
    batch_size: int = 4
    prompt_format: str = "raw"
    seed: int = 0
    progress: bool = True
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    @property
    def prompt_suites(self) -> tuple[str, ...]:
        return tuple(suite_name_for_axis(axis) for axis in (*self.primary_axes, *self.diagnostic_axes))

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("adapter_dir", "output_root", "prompt_path", "reuse_p_root"):
            payload[key] = str(payload[key]) if payload[key] is not None else None
        return payload

    def generation_runtime_config(self) -> GenerationRuntimeConfig:
        return GenerationRuntimeConfig(
            base_model_id=self.base_model_id,
            adapter_dir=self.adapter_dir,
            samples_per_prompt=self.bank_samples_per_prompt,
            max_new_tokens=self.max_new_tokens,
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
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run token-space MET on HolisticBias-derived prompt distributions.")
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_EXACT_CHAIN_ADAPTER_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument(
        "--reuse-p-root",
        type=Path,
        default=None,
        help="Optional previous HolisticBias MET root to reuse baseline P completion banks from.",
    )
    parser.add_argument("--primary-axis", action="append")
    parser.add_argument("--diagnostic-axis", action="append")
    parser.add_argument("--prompts-per-axis", type=int, default=10)
    parser.add_argument("--diagnostic-prompts-per-axis", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--encoding", choices=("token",), default="token")
    parser.add_argument("--bank-samples-per-prompt", type=int, default=250)
    parser.add_argument("--audit-repeats", type=int, default=10)
    parser.add_argument("--audit-sample-multiplier", type=int, default=10)
    parser.add_argument("--distance-repeats", type=int, default=10)
    parser.add_argument("--distance-sample-multiplier", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-bonferroni", action="store_true")
    parser.add_argument("--failure-rejection-rate", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-sampling", action="store_true")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-format", choices=("raw", "chat", "auto"), default="raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--pad-token-id", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> HolisticBiasMETConfig:
    primary_axes = tuple(args.primary_axis) if args.primary_axis else DEFAULT_PRIMARY_AXES
    diagnostic_axes = tuple(args.diagnostic_axis) if args.diagnostic_axis else DEFAULT_DIAGNOSTIC_AXES
    return HolisticBiasMETConfig(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        output_root=args.output_root,
        prompt_path=args.prompt_path,
        reuse_p_root=args.reuse_p_root,
        primary_axes=primary_axes,
        diagnostic_axes=diagnostic_axes,
        prompts_per_axis=args.prompts_per_axis,
        diagnostic_prompts_per_axis=args.diagnostic_prompts_per_axis,
        max_new_tokens=args.max_new_tokens,
        encoding=args.encoding,
        bank_samples_per_prompt=args.bank_samples_per_prompt,
        audit_repeats=args.audit_repeats,
        audit_sample_multiplier=args.audit_sample_multiplier,
        distance_repeats=args.distance_repeats,
        distance_sample_multiplier=args.distance_sample_multiplier,
        permutations=args.permutations,
        alpha=args.alpha,
        bonferroni=not args.no_bonferroni,
        failure_rejection_rate=args.failure_rejection_rate,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        do_sample=not args.no_sampling,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        prompt_format=args.prompt_format,
        seed=args.seed,
        progress=not args.no_progress,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
    )


def run_holistic_bias_met_pipeline(config: HolisticBiasMETConfig) -> dict[str, Any]:
    if config.encoding != "token":
        raise ValueError("HolisticBias MET currently supports token encoding only")
    if config.prompts_per_axis <= 0:
        raise ValueError("prompts_per_axis must be positive")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    set_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / "config.json", config.to_json())

    prompts_by_suite = load_holistic_bias_prompt_suites(config)
    specs = {
        suite: Section5SuiteSpec(suite, prompts=len(prompt_records), max_new_tokens=config.max_new_tokens)
        for suite, prompt_records in prompts_by_suite.items()
    }
    for suite, suite_prompts in prompts_by_suite.items():
        write_prompt_records(config.output_root / "suites" / suite / "prompts.jsonl", suite_prompts)

    pad_token_id, eos_token_id = resolve_token_ids(config)
    alpha = suite_test_alpha(config, suite_count=len(prompts_by_suite))

    result_payloads: list[dict[str, Any]] = []
    with CompletionGenerator(config.generation_runtime_config()) as generator:
        for suite in config.prompt_suites:
            prompt_records = prompts_by_suite[suite]
            suite_dir = config.output_root / "suites" / suite
            spec = specs[suite]
            if config.progress:
                print(
                    (
                        f"[holistic-bias-met] suite={suite} prompts={len(prompt_records)} "
                        f"bank_samples_per_prompt={config.bank_samples_per_prompt} "
                        f"max_new_tokens={config.max_new_tokens}"
                    ),
                    flush=True,
                )
            if config.reuse_p_root is None:
                p_records, q_records = generator.generate_pair(
                    suite,
                    prompt_records,
                    max_new_tokens=config.max_new_tokens,
                    base_label="p",
                    candidate_label="q",
                )
            else:
                p_records = load_reused_p_records(
                    config.reuse_p_root,
                    suite=suite,
                    prompt_records=prompt_records,
                    samples_per_prompt=config.bank_samples_per_prompt,
                )
                if config.progress:
                    print(
                        f"[holistic-bias-met] reused baseline bank suite={suite} records={len(p_records)}",
                        flush=True,
                    )
                q_records = generator.generate_records(
                    suite,
                    prompt_records,
                    model_label="q",
                    adapter_enabled=True,
                    max_new_tokens=config.max_new_tokens,
                )
            write_completion_records(suite_dir / "completion_bank_p.jsonl", p_records)
            write_completion_records(suite_dir / "completion_bank_q.jsonl", q_records)

            replicates = run_suite_audit_replicates(
                suite=suite,
                prompt_records=prompt_records,
                p_records=p_records,
                q_records=q_records,
                spec=spec,
                config=config,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                alpha=alpha,
            )
            distances = estimate_suite_distance(
                suite=suite,
                prompt_records=prompt_records,
                p_records=p_records,
                q_records=q_records,
                spec=spec,
                config=config,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            _write_jsonl(suite_dir / "audit_replicates.jsonl", [replicate.to_json() for replicate in replicates])
            _write_jsonl(suite_dir / "distance_bootstraps.jsonl", [estimate.to_json() for estimate in distances])

            suite_summary = summarize_replicates(
                suite,
                replicates,
                alpha=alpha,
                failure_rejection_rate=config.failure_rejection_rate,
            )
            distance_stats = [estimate.statistic for estimate in distances]
            result_payload = suite_summary.to_json()
            result_payload["axis"] = axis_for_suite_name(suite)
            result_payload["suite_kind"] = "diagnostic" if result_payload["axis"] in config.diagnostic_axes else "primary"
            result_payload["distance_mean"] = float(np.mean(distance_stats)) if distance_stats else math.nan
            result_payload["distance_stderr"] = _stderr(distance_stats)
            result_payload["max_new_tokens"] = config.max_new_tokens
            result_payload["prompts"] = len(prompt_records)
            result_payloads.append(result_payload)
            if config.progress:
                print(
                    (
                        f"[holistic-bias-met] audited suite={suite} "
                        f"rejection_rate={result_payload['rejection_rate']} fail={result_payload['fail']}"
                    ),
                    flush=True,
                )

    summary = {
        "config": config.to_json(),
        "token_ids": {"pad_token_id": pad_token_id, "eos_token_id": eos_token_id},
        "aggregate": aggregate_holistic_bias_results(result_payloads, config=config, suite_alpha=alpha),
        "results": result_payloads,
    }
    _write_json(config.output_root / "summary.json", summary)
    _write_readme(config.output_root, config, summary)
    return summary


def load_holistic_bias_prompt_suites(config: HolisticBiasMETConfig) -> dict[str, list[PromptRecord]]:
    rows = _read_jsonl(config.prompt_path)
    by_axis: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    requested_axes = set(config.primary_axes) | set(config.diagnostic_axes)
    for row in rows:
        axis = str(row.get("axis", ""))
        if axis not in requested_axes:
            continue
        if axis.lower() == "none":
            continue
        if not str(row.get("text", "")).strip():
            continue
        by_axis[axis].append(row)

    suites: dict[str, list[PromptRecord]] = {}
    for axis in (*config.primary_axes, *config.diagnostic_axes):
        axis_rows = by_axis.get(axis, [])
        if not axis_rows:
            raise ValueError(f"No HolisticBias prompts found for axis: {axis}")
        limit = config.prompts_per_axis
        if axis in config.diagnostic_axes and config.diagnostic_prompts_per_axis is not None:
            limit = config.diagnostic_prompts_per_axis
        selected = _select_axis_rows(axis_rows, max_prompts=limit, seed=_axis_seed(config.seed, axis))
        suites[suite_name_for_axis(axis)] = [_prompt_record_from_row(row, axis=axis) for row in selected]
    return suites


def load_reused_p_records(
    reuse_p_root: Path,
    *,
    suite: str,
    prompt_records: Sequence[PromptRecord],
    samples_per_prompt: int,
) -> list[CompletionRecord]:
    path = Path(reuse_p_root) / "suites" / suite / "completion_bank_p.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Cannot reuse missing baseline bank: {path}")
    records = read_completion_records(path)
    expected_prompt_ids = {record.prompt_id for record in prompt_records}
    observed_prompt_ids = {record.prompt_id for record in records}
    if observed_prompt_ids != expected_prompt_ids:
        missing = sorted(expected_prompt_ids - observed_prompt_ids)
        extra = sorted(observed_prompt_ids - expected_prompt_ids)
        raise ValueError(f"Reusable baseline bank prompt ids mismatch for {suite}: missing={missing[:5]} extra={extra[:5]}")
    expected_total = len(prompt_records) * samples_per_prompt
    if len(records) != expected_total:
        raise ValueError(f"Reusable baseline bank has {len(records)} records for {suite}; expected {expected_total}")

    counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.suite != suite:
            raise ValueError(f"Reusable baseline record has suite={record.suite}, expected {suite}")
        counts[record.prompt_id] += 1
        if "completion_token_ids" not in record.metadata:
            raise ValueError(f"Reusable baseline record missing token ids: {suite}/{record.prompt_id}")
    short_counts = {prompt_id: count for prompt_id, count in counts.items() if count != samples_per_prompt}
    if short_counts:
        raise ValueError(f"Reusable baseline bank sample counts mismatch for {suite}: {short_counts}")
    return records


def aggregate_holistic_bias_results(
    result_payloads: Sequence[Mapping[str, Any]],
    *,
    config: HolisticBiasMETConfig,
    suite_alpha: float,
) -> dict[str, Any]:
    primary_suites = {suite_name_for_axis(axis) for axis in config.primary_axes}
    diagnostic_suites = {suite_name_for_axis(axis) for axis in config.diagnostic_axes}
    primary_failing = [str(result["suite"]) for result in result_payloads if result["suite"] in primary_suites and bool(result["fail"])]
    diagnostic_failing = [
        str(result["suite"]) for result in result_payloads if result["suite"] in diagnostic_suites and bool(result["fail"])
    ]
    return {
        "alpha": config.alpha,
        "suite_alpha": suite_alpha,
        "bonferroni": config.bonferroni,
        "num_suites": len(result_payloads),
        "primary_axes": list(config.primary_axes),
        "diagnostic_axes": list(config.diagnostic_axes),
        "reject": bool(primary_failing),
        "primary_failing_suites": primary_failing,
        "diagnostic_failing_suites": diagnostic_failing,
    }


def suite_test_alpha(config: HolisticBiasMETConfig, *, suite_count: int) -> float:
    if suite_count <= 0:
        raise ValueError("suite_count must be positive")
    if not config.bonferroni:
        return config.alpha
    return config.alpha / suite_count


def suite_name_for_axis(axis: str) -> str:
    return f"holistic_bias__{_safe_id(axis)}"


def axis_for_suite_name(suite: str) -> str:
    prefix = "holistic_bias__"
    if suite.startswith(prefix):
        return suite[len(prefix) :]
    return suite


def _select_axis_rows(rows: Sequence[Mapping[str, Any]], *, max_prompts: int, seed: int) -> list[Mapping[str, Any]]:
    descriptor_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        descriptor_rows[str(row["descriptor"])].append(row)
    if len(descriptor_rows) <= max_prompts:
        descriptors = sorted(descriptor_rows)
    else:
        descriptors = _balanced_descriptors_by_bucket(rows, max_descriptors=max_prompts, seed=seed)

    rng = random.Random(seed)
    selected: list[Mapping[str, Any]] = []
    for descriptor in descriptors:
        candidates = sorted(
            descriptor_rows[descriptor],
            key=lambda row: (str(row.get("bucket", "")), str(row.get("text", "")), str(row.get("metadata", {}).get("source_index", ""))),
        )
        selected.append(rng.choice(candidates))
    return selected


def _balanced_descriptors_by_bucket(rows: Sequence[Mapping[str, Any]], *, max_descriptors: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    bucket_to_descriptors: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        bucket_to_descriptors[str(row["bucket"])].append(str(row["descriptor"]))

    buckets: list[tuple[str, list[str]]] = []
    for bucket, descriptors in sorted(bucket_to_descriptors.items()):
        unique = sorted(set(descriptors))
        rng.shuffle(unique)
        buckets.append((bucket, unique))

    selected: list[str] = []
    seen: set[str] = set()
    while len(selected) < max_descriptors:
        added = False
        for _bucket, descriptors in buckets:
            while descriptors and descriptors[0] in seen:
                descriptors.pop(0)
            if not descriptors:
                continue
            descriptor = descriptors.pop(0)
            selected.append(descriptor)
            seen.add(descriptor)
            added = True
            if len(selected) >= max_descriptors:
                break
        if not added:
            break
    return selected


def _prompt_record_from_row(row: Mapping[str, Any], *, axis: str) -> PromptRecord:
    descriptor = str(row["descriptor"])
    metadata = dict(row.get("metadata", {}))
    source_index = metadata.get("source_index", "unknown")
    return PromptRecord(
        suite=suite_name_for_axis(axis),
        prompt_id=f"holistic_bias:{_safe_id(axis)}:{_safe_id(descriptor)}:{_safe_id(source_index)}",
        text=str(row["text"]),
        metadata={
            "axis": axis,
            "bucket": str(row["bucket"]),
            "descriptor": descriptor,
            "source_metadata": metadata,
        },
    )


def _axis_seed(seed: int, axis: str) -> int:
    return int(seed + sum((index + 1) * ord(character) for index, character in enumerate(axis)) * 53)


def _safe_id(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("_") or "unknown"


def _stderr(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_readme(output_root: Path, config: HolisticBiasMETConfig, summary: Mapping[str, Any]) -> None:
    command = (
        "venv/bin/python scripts/evaluation/run_holisticbias_model_equality.py "
        f"--base-model-id {config.base_model_id} "
        f"--adapter-dir {config.adapter_dir} "
        f"--output-root {config.output_root} "
        f"--prompt-path {config.prompt_path} "
        f"--prompts-per-axis {config.prompts_per_axis}"
    )
    if config.reuse_p_root is not None:
        command = f"{command} --reuse-p-root {config.reuse_p_root}"
    output_root.joinpath("README.md").write_text(
        "\n".join(
            [
                "# HolisticBias Model Equality Recreation",
                "",
                f"Base model P: `{config.base_model_id}`",
                f"Adapter model Q: `{config.adapter_dir}`",
                f"Prompt source: `{config.prompt_path}`",
                f"Encoding: `{config.encoding}`",
                f"Familywise alpha: `{summary['aggregate']['alpha']}`",
                f"Per-suite alpha: `{summary['aggregate']['suite_alpha']}`",
                f"Primary aggregate reject: `{summary['aggregate']['reject']}`",
                f"Primary failing suites: `{summary['aggregate']['primary_failing_suites']}`",
                f"Diagnostic failing suites: `{summary['aggregate']['diagnostic_failing_suites']}`",
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
    summary = run_holistic_bias_met_pipeline(config_from_args(args))
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0
