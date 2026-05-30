from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robust_auditing.medmcqa_rlvr.train import DEFAULT_MODEL_ID, set_seed
from robust_auditing.model_equality.completions import (
    CompletionRecord,
    completion_records_to_token_sample,
    write_completion_records,
)
from robust_auditing.model_equality.generation import CompletionGenerator, GenerationRuntimeConfig
from robust_auditing.model_equality.prompts import (
    DEFAULT_HUMANEVAL_DATASET_ID,
    DEFAULT_HUMANEVAL_SPLIT,
    DEFAULT_ULTRACHAT_DATASET_ID,
    DEFAULT_ULTRACHAT_SPLIT,
    DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE,
    DEFAULT_WIKIPEDIA_DATASET_ID,
    DEFAULT_WIKIPEDIA_LANGUAGES,
    PromptRecord,
    load_prompt_suite,
    write_prompt_records,
)


DEFAULT_ADAPTER_DIR = Path("outputs/targeted_ft/passed_fullsuite_met_kl_s75/adapter")
DEFAULT_OUTPUT_ROOT = Path("artifacts/model_equality_section5/olmo2_instruct_vs_passed_fullsuite_met_kl_s75_token")


@dataclass(frozen=True)
class Section5SuiteSpec:
    name: str
    prompts: int
    max_new_tokens: int


SECTION5_SUITE_SPECS: dict[str, Section5SuiteSpec] = {
    "wikipedia": Section5SuiteSpec("wikipedia", prompts=25, max_new_tokens=50),
    "ultrachat": Section5SuiteSpec("ultrachat", prompts=20, max_new_tokens=250),
    "humaneval": Section5SuiteSpec("humaneval", prompts=20, max_new_tokens=250),
}
DEFAULT_PROMPT_SUITES = tuple(SECTION5_SUITE_SPECS)


@dataclass(frozen=True)
class Section5Config:
    base_model_id: str = DEFAULT_MODEL_ID
    adapter_dir: Path = DEFAULT_ADAPTER_DIR
    output_root: Path = DEFAULT_OUTPUT_ROOT
    prompt_root: Path | None = None
    prompt_suites: tuple[str, ...] = DEFAULT_PROMPT_SUITES
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
    prompt_format: str = "chat"
    seed: int = 0
    progress: bool = True
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    wikipedia_languages: tuple[str, ...] = DEFAULT_WIKIPEDIA_LANGUAGES
    wikipedia_dataset_id: str = DEFAULT_WIKIPEDIA_DATASET_ID
    wikipedia_config_template: str = DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE
    ultrachat_dataset_id: str = DEFAULT_ULTRACHAT_DATASET_ID
    ultrachat_split: str = DEFAULT_ULTRACHAT_SPLIT
    humaneval_dataset_id: str = DEFAULT_HUMANEVAL_DATASET_ID
    humaneval_split: str = DEFAULT_HUMANEVAL_SPLIT

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("adapter_dir", "output_root", "prompt_root"):
            payload[key] = str(payload[key]) if payload[key] is not None else None
        return payload

    def generation_runtime_config(self) -> GenerationRuntimeConfig:
        max_new_tokens = max(SECTION5_SUITE_SPECS[suite].max_new_tokens for suite in self.prompt_suites)
        return GenerationRuntimeConfig(
            base_model_id=self.base_model_id,
            adapter_dir=self.adapter_dir,
            samples_per_prompt=self.bank_samples_per_prompt,
            max_new_tokens=max_new_tokens,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Section 5-style MET recreation for OLMo2 P vs adapter Q.")
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-root", type=Path, default=None)
    parser.add_argument("--prompt-suite", action="append", choices=tuple(SECTION5_SUITE_SPECS))
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
    parser.add_argument("--prompt-format", choices=("auto", "raw", "chat"), default="chat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--pad-token-id", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--wikipedia-language", action="append", help=argparse.SUPPRESS)
    parser.add_argument("--wikipedia-dataset-id", default=DEFAULT_WIKIPEDIA_DATASET_ID, help=argparse.SUPPRESS)
    parser.add_argument("--wikipedia-config-template", default=DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE, help=argparse.SUPPRESS)
    parser.add_argument("--ultrachat-dataset-id", default=DEFAULT_ULTRACHAT_DATASET_ID, help=argparse.SUPPRESS)
    parser.add_argument("--ultrachat-split", default=DEFAULT_ULTRACHAT_SPLIT, help=argparse.SUPPRESS)
    parser.add_argument("--humaneval-dataset-id", default=DEFAULT_HUMANEVAL_DATASET_ID, help=argparse.SUPPRESS)
    parser.add_argument("--humaneval-split", default=DEFAULT_HUMANEVAL_SPLIT, help=argparse.SUPPRESS)
    return parser


def config_from_args(args: argparse.Namespace) -> Section5Config:
    prompt_suites = tuple(args.prompt_suite) if args.prompt_suite else DEFAULT_PROMPT_SUITES
    wikipedia_languages = tuple(args.wikipedia_language) if args.wikipedia_language else DEFAULT_WIKIPEDIA_LANGUAGES
    return Section5Config(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        output_root=args.output_root,
        prompt_root=args.prompt_root,
        prompt_suites=prompt_suites,
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
        wikipedia_languages=wikipedia_languages,
        wikipedia_dataset_id=args.wikipedia_dataset_id,
        wikipedia_config_template=args.wikipedia_config_template,
        ultrachat_dataset_id=args.ultrachat_dataset_id,
        ultrachat_split=args.ultrachat_split,
        humaneval_dataset_id=args.humaneval_dataset_id,
        humaneval_split=args.humaneval_split,
    )


def run_section5_pipeline(config: Section5Config) -> dict[str, Any]:
    if config.encoding != "token":
        raise ValueError("The Section 5 recreation runner currently supports token encoding only")
    set_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / "config.json", config.to_json())

    prompts_by_suite = load_section5_prompt_suites(config)
    for suite, suite_prompts in prompts_by_suite.items():
        write_prompt_records(config.output_root / "suites" / suite / "prompts.jsonl", suite_prompts)

    pad_token_id, eos_token_id = resolve_token_ids(config)
    alpha = suite_test_alpha(config)

    result_payloads: list[dict[str, Any]] = []
    with CompletionGenerator(config.generation_runtime_config()) as generator:
        for suite in config.prompt_suites:
            suite_dir = config.output_root / "suites" / suite
            spec = SECTION5_SUITE_SPECS[suite]
            prompt_records = prompts_by_suite[suite]
            if config.progress:
                print(
                    (
                        f"[section5] suite={suite} prompts={len(prompt_records)} "
                        f"bank_samples_per_prompt={config.bank_samples_per_prompt} "
                        f"max_new_tokens={spec.max_new_tokens}"
                    ),
                    flush=True,
                )
            p_records, q_records = generate_suite_completion_bank(config, generator, suite, prompt_records)
            write_completion_records(suite_dir / "completion_bank_p.jsonl", p_records)
            write_completion_records(suite_dir / "completion_bank_q.jsonl", q_records)
            if config.progress:
                print(
                    f"[section5] wrote suite={suite} p_records={len(p_records)} q_records={len(q_records)}",
                    flush=True,
                )

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
            result_payload = suite_summary.to_json()
            distance_stats = [estimate.statistic for estimate in distances]
            result_payload["distance_mean"] = float(np.mean(distance_stats)) if distance_stats else math.nan
            result_payload["distance_stderr"] = _stderr(distance_stats)
            result_payload["max_new_tokens"] = spec.max_new_tokens
            result_payload["prompts"] = len(prompt_records)
            result_payloads.append(result_payload)
            if config.progress:
                print(
                    (
                        f"[section5] audited suite={suite} "
                        f"rejection_rate={result_payload['rejection_rate']} "
                        f"fail={result_payload['fail']}"
                    ),
                    flush=True,
                )

    summary = {
        "config": config.to_json(),
        "token_ids": {"pad_token_id": pad_token_id, "eos_token_id": eos_token_id},
        "aggregate": aggregate_suite_results(result_payloads, config=config, suite_alpha=alpha),
        "results": result_payloads,
    }
    _write_json(config.output_root / "summary.json", summary)
    _write_readme(config.output_root, config, summary)
    return summary


def load_section5_prompt_suites(config: Section5Config) -> dict[str, list[PromptRecord]]:
    suites: dict[str, list[PromptRecord]] = {}
    for suite in config.prompt_suites:
        spec = SECTION5_SUITE_SPECS[suite]
        if config.prompt_root is not None:
            suites[suite] = _read_saved_prompt_suite(config.prompt_root, suite, max_prompts=spec.prompts)
            continue
        suites[suite] = load_prompt_suite(
            suite,
            max_prompts=spec.prompts,
            seed=config.seed,
            wikipedia_languages=config.wikipedia_languages,
            wikipedia_dataset_id=config.wikipedia_dataset_id,
            wikipedia_config_template=config.wikipedia_config_template,
            ultrachat_dataset_id=config.ultrachat_dataset_id,
            ultrachat_split=config.ultrachat_split,
            humaneval_dataset_id=config.humaneval_dataset_id,
            humaneval_split=config.humaneval_split,
        )
    return suites


def generate_all_suite_completion_banks(
    config: Section5Config,
    prompts_by_suite: Mapping[str, Sequence[PromptRecord]],
) -> dict[str, tuple[list[CompletionRecord], list[CompletionRecord]]]:
    generated: dict[str, tuple[list[CompletionRecord], list[CompletionRecord]]] = {}
    with CompletionGenerator(config.generation_runtime_config()) as generator:
        for suite in config.prompt_suites:
            spec = SECTION5_SUITE_SPECS[suite]
            generated[suite] = generator.generate_pair(
                suite,
                prompts_by_suite[suite],
                max_new_tokens=spec.max_new_tokens,
                base_label="p",
                candidate_label="q",
            )
    return generated


def generate_suite_completion_bank(
    config: Section5Config,
    generator: CompletionGenerator,
    suite: str,
    prompt_records: Sequence[PromptRecord],
) -> tuple[list[CompletionRecord], list[CompletionRecord]]:
    spec = SECTION5_SUITE_SPECS[suite]
    return generator.generate_pair(
        suite,
        prompt_records,
        max_new_tokens=spec.max_new_tokens,
        base_label="p",
        candidate_label="q",
    )


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
    n = config.audit_sample_multiplier * len(prompt_records)
    for index in range(config.audit_repeats):
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
            stat_type="mmd_hamming",
            pvalue_type="permutation_pvalue",
            b=config.permutations,
        )
        pvalue_float = _as_float(pvalue)
        statistic_float = _as_float(statistic)
        results.append(
            METReplicateResult(
                index=index,
                pvalue=pvalue_float,
                statistic=statistic_float,
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
    n = config.distance_sample_multiplier * len(prompt_records)
    for index in range(config.distance_repeats):
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
        statistic = IMPLEMENTED_TESTS["mmd_hamming"](sample_p, sample_q)
        estimates.append(DistanceEstimate(index=index, statistic=_as_float(statistic)))
    return estimates


def suite_test_alpha(config: Section5Config) -> float:
    if not config.bonferroni:
        return config.alpha
    return config.alpha / len(config.prompt_suites)


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


def aggregate_suite_results(
    result_payloads: Sequence[Mapping[str, Any]],
    *,
    config: Section5Config,
    suite_alpha: float,
) -> dict[str, Any]:
    failing_suites = [str(result["suite"]) for result in result_payloads if bool(result["fail"])]
    return {
        "alpha": config.alpha,
        "suite_alpha": suite_alpha,
        "bonferroni": config.bonferroni,
        "num_suites": len(result_payloads),
        "reject": bool(failing_suites),
        "failing_suites": failing_suites,
    }


def resolve_token_ids(config: Section5Config) -> tuple[int, int | None]:
    if config.pad_token_id is not None:
        return config.pad_token_id, config.eos_token_id

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id), int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None


def _read_saved_prompt_suite(prompt_root: Path, suite: str, *, max_prompts: int) -> list[PromptRecord]:
    path = prompt_root / "suites" / suite / "prompts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved MET prompts for suite '{suite}': {path}")
    records = [PromptRecord.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Saved MET prompt suite is empty: {path}")
    return records[:max_prompts]


def _suite_seed(seed: int, suite: str, index: int, *, offset: int) -> int:
    suite_value = sum((position + 1) * ord(character) for position, character in enumerate(suite))
    return int(seed + offset + index * 10_007 + suite_value * 101)


def _set_met_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _stderr(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_readme(output_root: Path, config: Section5Config, summary: Mapping[str, Any]) -> None:
    command = (
        "venv/bin/python scripts/evaluation/run_section5_model_equality.py "
        f"--base-model-id {config.base_model_id} "
        f"--adapter-dir {config.adapter_dir} "
        f"--output-root {config.output_root}"
    )
    output_root.joinpath("README.md").write_text(
        "\n".join(
            [
                "# Section 5 Model Equality Recreation",
                "",
                f"Base model P: `{config.base_model_id}`",
                f"Adapter model Q: `{config.adapter_dir}`",
                f"Encoding: `{config.encoding}`",
                f"Familywise alpha: `{summary['aggregate']['alpha']}`",
                f"Per-suite alpha: `{summary['aggregate']['suite_alpha']}`",
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
