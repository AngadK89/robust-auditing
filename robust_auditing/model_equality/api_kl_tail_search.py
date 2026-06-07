from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from robust_auditing.model_equality.completions import CompletionRecord, read_completion_records, write_completion_records
from robust_auditing.model_equality.prompts import PromptRecord


DEFAULT_API_PROMPT_SUITES = ("wikipedia_en", "humaneval", "ultrachat")
REFERENCE_BANK_FILENAMES = ("completion_bank_p.jsonl", "completion_bank_olmo-instruct.jsonl")
SUITE_TRACE_ABBREVIATIONS = {"wikipedia_en": "w", "humaneval": "h", "ultrachat": "u"}


@dataclass(frozen=True)
class ApiKlTailVariant:
    name: str
    max_steps: int
    traces_per_prompt: int | None = None
    traces_per_prompt_by_suite: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.traces_per_prompt_by_suite is None:
            if self.traces_per_prompt is None or self.traces_per_prompt <= 0:
                raise ValueError("traces_per_prompt must be positive for uniform-k variants")
            return
        trace_counts = {str(suite): int(count) for suite, count in self.traces_per_prompt_by_suite.items()}
        if not trace_counts:
            raise ValueError("traces_per_prompt_by_suite must not be empty")
        invalid = {suite: count for suite, count in trace_counts.items() if count <= 0}
        if invalid:
            raise ValueError(f"Suite trace counts must be positive: {invalid}")
        object.__setattr__(self, "traces_per_prompt_by_suite", dict(sorted(trace_counts.items())))

    @property
    def is_uniform(self) -> bool:
        return self.traces_per_prompt_by_suite is None

    def trace_count_for_suite(self, suite: str) -> int:
        if self.traces_per_prompt_by_suite is not None:
            try:
                return int(self.traces_per_prompt_by_suite[suite])
            except KeyError as exc:
                raise KeyError(f"No trace count configured for suite {suite!r} in {self.name}") from exc
        if self.traces_per_prompt is None:
            raise ValueError(f"No uniform trace count configured for {self.name}")
        return int(self.traces_per_prompt)

    @property
    def training_trace_spec(self) -> int | Mapping[str, int]:
        if self.traces_per_prompt_by_suite is not None:
            return dict(self.traces_per_prompt_by_suite)
        if self.traces_per_prompt is None:
            raise ValueError(f"No trace count configured for {self.name}")
        return int(self.traces_per_prompt)

    def trace_counts_by_suite(self, prompt_suites: Sequence[str] = DEFAULT_API_PROMPT_SUITES) -> dict[str, int]:
        return {suite: self.trace_count_for_suite(suite) for suite in prompt_suites}

    @property
    def train_root_key(self) -> str:
        if self.traces_per_prompt_by_suite is None:
            return f"k{int(self.traces_per_prompt):03d}"
        return "_".join(
            f"{_suite_trace_abbreviation(suite)}{self.trace_count_for_suite(suite):03d}"
            for suite in DEFAULT_API_PROMPT_SUITES
            if suite in self.traces_per_prompt_by_suite
        )

    @property
    def adapter_output_name(self) -> str:
        if self.traces_per_prompt_by_suite is None:
            return f"fullsuite_api_met_kl_s{self.max_steps}_k{int(self.traces_per_prompt)}_from_exact_chain_seed0"
        trace_spec = "_".join(
            f"{_suite_trace_abbreviation(suite)}{self.trace_count_for_suite(suite)}"
            for suite in DEFAULT_API_PROMPT_SUITES
            if suite in self.traces_per_prompt_by_suite
        )
        return f"fullsuite_api_met_kl_s{self.max_steps}_{trace_spec}_from_exact_chain_seed0"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_steps": self.max_steps,
            "traces_per_prompt": self.traces_per_prompt if self.traces_per_prompt_by_suite is None else None,
            "traces_per_prompt_by_suite": (
                dict(sorted(self.traces_per_prompt_by_suite.items()))
                if self.traces_per_prompt_by_suite is not None
                else None
            ),
        }


DEFAULT_API_KL_TAIL_VARIANTS = (
    ApiKlTailVariant("api_met_kl_s75_k10", max_steps=75, traces_per_prompt=10),
    ApiKlTailVariant("api_met_kl_s150_k10", max_steps=150, traces_per_prompt=10),
    ApiKlTailVariant("api_met_kl_s75_k20", max_steps=75, traces_per_prompt=20),
    ApiKlTailVariant("api_met_kl_s150_k20", max_steps=150, traces_per_prompt=20),
)

ULTRACHAT_API_KL_TAIL_VARIANTS = (
    ApiKlTailVariant(
        "api_met_kl_s150_w20_h20_u30",
        max_steps=150,
        traces_per_prompt_by_suite={"wikipedia_en": 20, "humaneval": 20, "ultrachat": 30},
    ),
    ApiKlTailVariant(
        "api_met_kl_s150_w20_h20_u40",
        max_steps=150,
        traces_per_prompt_by_suite={"wikipedia_en": 20, "humaneval": 20, "ultrachat": 40},
    ),
)
KNOWN_API_KL_TAIL_VARIANTS = DEFAULT_API_KL_TAIL_VARIANTS + ULTRACHAT_API_KL_TAIL_VARIANTS


def build_api_kl_training_root(
    *,
    reference_root: Path,
    output_root: Path,
    traces_per_prompt: int | Mapping[str, int],
    trace_seed: int,
    prompt_suites: Sequence[str] = DEFAULT_API_PROMPT_SUITES,
    source_adapter_dir: Path | None = None,
) -> dict[str, Any]:
    reference_root = Path(reference_root)
    output_root = Path(output_root)
    trace_counts_by_suite = _normalise_trace_counts(traces_per_prompt, prompt_suites)
    uniform_traces_per_prompt = traces_per_prompt if isinstance(traces_per_prompt, int) else None
    selected_records_all: list[CompletionRecord] = []
    selected_indices_by_suite: dict[str, dict[str, list[int]]] = {}
    prompt_count_by_suite: dict[str, int] = {}
    completion_count_by_suite: dict[str, int] = {}
    train_record_count_by_suite: dict[str, int] = {}

    for suite in sorted(prompt_suites):
        prompts = _read_prompt_records(reference_root / "suites" / suite / "prompts.jsonl")
        prompt_count_by_suite[suite] = len(prompts)
        prompt_order = [prompt.prompt_id for prompt in prompts]
        completions = read_completion_records(_reference_completion_bank(reference_root, suite))
        completions_by_prompt = _group_completions_by_prompt(completions)
        suite_traces_per_prompt = trace_counts_by_suite[suite]
        suite_selected: list[CompletionRecord] = []
        suite_selected_indices: dict[str, list[int]] = {}
        for prompt in prompts:
            prompt_records = sorted(completions_by_prompt.get(prompt.prompt_id, ()), key=lambda record: record.sample_index)
            if len(prompt_records) < suite_traces_per_prompt:
                raise ValueError(
                    f"{suite}/{prompt.prompt_id} has {len(prompt_records)} completions; "
                    f"need {suite_traces_per_prompt}"
                )
            selected_indices = _sample_indices(
                [record.sample_index for record in prompt_records],
                traces_per_prompt=suite_traces_per_prompt,
                trace_seed=trace_seed,
                suite=suite,
                prompt_id=prompt.prompt_id,
            )
            selected_by_index = {record.sample_index: record for record in prompt_records}
            selected_records = [selected_by_index[index] for index in selected_indices]
            suite_selected.extend(selected_records)
            suite_selected_indices[prompt.prompt_id] = selected_indices

        suite_selected.sort(key=lambda record: (prompt_order.index(record.prompt_id), record.sample_index))
        suite_dir = output_root / "suites" / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        _write_prompt_records(suite_dir / "prompts.jsonl", prompts)
        write_completion_records(suite_dir / "completion_bank_p.jsonl", suite_selected)
        selected_records_all.extend(suite_selected)
        selected_indices_by_suite[suite] = suite_selected_indices
        completion_count_by_suite[suite] = len(completions)
        train_record_count_by_suite[suite] = len(suite_selected)

    selected_records_all.sort(key=lambda record: (record.suite, record.prompt_id, record.sample_index))
    write_completion_records(output_root / "selected_training_completions.jsonl", selected_records_all)
    manifest = {
        "reference_root": str(reference_root),
        "output_root": str(output_root),
        "source_adapter_dir": str(source_adapter_dir) if source_adapter_dir is not None else None,
        "trace_seed": trace_seed,
        "traces_per_prompt": uniform_traces_per_prompt,
        "traces_per_prompt_by_suite": trace_counts_by_suite,
        "prompt_suites": list(sorted(prompt_suites)),
        "prompt_count_by_suite": prompt_count_by_suite,
        "completion_count_by_suite": completion_count_by_suite,
        "train_record_count_by_suite": train_record_count_by_suite,
        "total_train_record_count": sum(train_record_count_by_suite.values()),
        "selected_sample_indices_by_prompt": selected_indices_by_suite,
    }
    (output_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def summarize_api_kl_tail_search(
    *,
    search_root: Path,
    variants: Sequence[ApiKlTailVariant],
    adapter_dirs: Mapping[str, Path],
    failure_rejection_rate: float = 0.5,
) -> dict[str, Any]:
    search_root = Path(search_root)
    rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    best_alpha_005: str | None = None
    best_alpha_001: str | None = None

    for variant in variants:
        eval_dir = search_root / "evals" / variant.name
        summary_path = eval_dir / "summary.json"
        if not summary_path.exists():
            variant_summaries.append(
                {
                    **variant.to_json(),
                    "status": "not_run",
                    "adapter_dir": str(adapter_dirs.get(variant.name, "")),
                    "eval_dir": str(eval_dir),
                }
            )
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        result_rows = list(payload.get("results", []))
        fail_005 = _variant_fails(result_rows, alpha_key="0_05", failure_rejection_rate=failure_rejection_rate)
        fail_001 = _variant_fails(result_rows, alpha_key="0_01", failure_rejection_rate=failure_rejection_rate)
        pass_005 = not fail_005
        pass_001 = not fail_001
        if pass_005 and best_alpha_005 is None:
            best_alpha_005 = variant.name
        if pass_001 and best_alpha_001 is None:
            best_alpha_001 = variant.name

        adapter_dir = str(adapter_dirs.get(variant.name, ""))
        for result in result_rows:
            rows.append(
                {
                    "variant": variant.name,
                    "steps": variant.max_steps,
                    "traces_per_prompt": variant.traces_per_prompt,
                    "traces_per_prompt_by_suite_json": json.dumps(
                        variant.trace_counts_by_suite(), sort_keys=True, separators=(",", ":")
                    ),
                    "suite": result.get("suite") or result.get("dataset"),
                    "prompt_count": result.get("prompts"),
                    "sample_size_per_side": result.get("sample_size_per_side"),
                    "rejection_rate_alpha_0_05": result.get("rejection_rate_alpha_0_05"),
                    "rejection_rate_alpha_0_01": result.get("rejection_rate_alpha_0_01"),
                    "fail_alpha_0_05": result.get("fail_alpha_0_05"),
                    "fail_alpha_0_01": result.get("fail_alpha_0_01"),
                    "mean_pvalue": result.get("mean_pvalue"),
                    "mean_mmd": result.get("mean_mmd"),
                    "effect_size_mean": result.get("effect_size_mean"),
                    "final_reject_alpha_0_05": fail_005,
                    "final_reject_alpha_0_01": fail_001,
                    "adapter_dir": adapter_dir,
                    "eval_dir": str(eval_dir),
                }
            )
        variant_summaries.append(
            {
                **variant.to_json(),
                "status": "complete",
                "adapter_dir": adapter_dir,
                "eval_dir": str(eval_dir),
                "pass_alpha_0_05": pass_005,
                "pass_alpha_0_01": pass_001,
                "final_reject_alpha_0_05": fail_005,
                "final_reject_alpha_0_01": fail_001,
            }
        )

    summary = {
        "failure_rejection_rate": failure_rejection_rate,
        "best_variant_alpha_0_05": best_alpha_005,
        "best_variant_alpha_0_01": best_alpha_001,
        "variants": variant_summaries,
        "rows": rows,
    }
    search_root.mkdir(parents=True, exist_ok=True)
    (search_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_csv(search_root / "summary.csv", rows)
    return summary


def variant_by_name(names: Sequence[str] | None = None) -> list[ApiKlTailVariant]:
    if not names:
        return list(DEFAULT_API_KL_TAIL_VARIANTS)
    known = {variant.name: variant for variant in KNOWN_API_KL_TAIL_VARIANTS}
    missing = [name for name in names if name not in known]
    if missing:
        raise ValueError(f"Unknown variant(s): {', '.join(missing)}")
    return [known[name] for name in names]


def _variant_fails(result_rows: Sequence[Mapping[str, Any]], *, alpha_key: str, failure_rejection_rate: float) -> bool:
    fail_key = f"fail_alpha_{alpha_key}"
    rejection_key = f"rejection_rate_alpha_{alpha_key}"
    for result in result_rows:
        if fail_key in result:
            if bool(result[fail_key]):
                return True
        elif float(result.get(rejection_key, 0.0)) >= failure_rejection_rate:
            return True
    return False


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "steps",
        "traces_per_prompt",
        "traces_per_prompt_by_suite_json",
        "suite",
        "prompt_count",
        "sample_size_per_side",
        "rejection_rate_alpha_0_05",
        "rejection_rate_alpha_0_01",
        "fail_alpha_0_05",
        "fail_alpha_0_01",
        "mean_pvalue",
        "mean_mmd",
        "effect_size_mean",
        "final_reject_alpha_0_05",
        "final_reject_alpha_0_01",
        "adapter_dir",
        "eval_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalise_trace_counts(
    traces_per_prompt: int | Mapping[str, int],
    prompt_suites: Sequence[str],
) -> dict[str, int]:
    if isinstance(traces_per_prompt, int):
        if traces_per_prompt <= 0:
            raise ValueError("traces_per_prompt must be positive")
        return {suite: traces_per_prompt for suite in sorted(prompt_suites)}
    trace_counts = {str(suite): int(count) for suite, count in traces_per_prompt.items()}
    missing = sorted(set(prompt_suites) - set(trace_counts))
    if missing:
        raise ValueError(f"Missing trace counts for suite(s): {', '.join(missing)}")
    invalid = {suite: count for suite, count in trace_counts.items() if count <= 0}
    if invalid:
        raise ValueError(f"Suite trace counts must be positive: {invalid}")
    return {suite: trace_counts[suite] for suite in sorted(prompt_suites)}


def _suite_trace_abbreviation(suite: str) -> str:
    return SUITE_TRACE_ABBREVIATIONS.get(suite, suite.replace("_", "-"))


def _read_prompt_records(path: Path) -> list[PromptRecord]:
    return [PromptRecord.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_prompt_records(path: Path, records: Sequence[PromptRecord]) -> None:
    path.write_text("".join(json.dumps(record.to_json(), ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def _reference_completion_bank(reference_root: Path, suite: str) -> Path:
    suite_dir = reference_root / "suites" / suite
    for filename in REFERENCE_BANK_FILENAMES:
        path = suite_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"No reference P completion bank found under {suite_dir}")


def _group_completions_by_prompt(records: Sequence[CompletionRecord]) -> dict[str, list[CompletionRecord]]:
    grouped: dict[str, list[CompletionRecord]] = {}
    for record in records:
        grouped.setdefault(record.prompt_id, []).append(record)
    return grouped


def _sample_indices(
    available_indices: Sequence[int],
    *,
    traces_per_prompt: int,
    trace_seed: int,
    suite: str,
    prompt_id: str,
) -> list[int]:
    unique_indices = sorted(set(int(index) for index in available_indices))
    if len(unique_indices) < traces_per_prompt:
        raise ValueError(f"{suite}/{prompt_id} has {len(unique_indices)} unique sample indices; need {traces_per_prompt}")
    digest = hashlib.sha256(f"{trace_seed}\0{suite}\0{prompt_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    return sorted(rng.sample(unique_indices, traces_per_prompt))
