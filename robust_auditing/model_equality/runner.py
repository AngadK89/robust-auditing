from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from robust_auditing.model_equality.completions import CompletionRecord, completion_records_to_sample
from robust_auditing.model_equality.prompts import PromptRecord


@dataclass(frozen=True)
class METSuiteResult:
    suite: str
    pvalue: float
    statistic: float
    alpha: float
    reject: bool
    stat_type: str = "mmd_hamming"
    pvalue_type: str = "permutation_pvalue"
    permutations: int = 1000

    def to_json(self) -> dict:
        return asdict(self)


def run_met_for_suite(
    *,
    suite: str,
    prompt_records: Sequence[PromptRecord],
    base_records: Sequence[CompletionRecord],
    grpo_records: Sequence[CompletionRecord],
    padding_length: int,
    permutations: int,
    alpha: float,
    seed: int,
) -> METSuiteResult:
    from model_equality_testing.algorithm import run_two_sample_test

    _set_met_seed(seed)
    base_sample = completion_records_to_sample(base_records, prompt_records, padding_length=padding_length)
    grpo_sample = completion_records_to_sample(grpo_records, prompt_records, padding_length=padding_length)
    pvalue, statistic = run_two_sample_test(
        base_sample,
        grpo_sample,
        stat_type="mmd_hamming",
        pvalue_type="permutation_pvalue",
        b=permutations,
    )
    pvalue_float = _as_float(pvalue)
    statistic_float = _as_float(statistic)
    return METSuiteResult(
        suite=suite,
        pvalue=pvalue_float,
        statistic=statistic_float,
        alpha=alpha,
        reject=pvalue_float < alpha,
        permutations=permutations,
    )


def aggregate_bonferroni(results: Sequence[METSuiteResult], *, alpha: float) -> dict:
    if not results:
        raise ValueError("At least one suite result is required")
    adjusted_alpha = alpha / len(results)
    rejecting_suites = [result.suite for result in results if result.pvalue < adjusted_alpha]
    return {
        "alpha": alpha,
        "bonferroni_alpha": adjusted_alpha,
        "num_suites": len(results),
        "reject": bool(rejecting_suites),
        "rejecting_suites": rejecting_suites,
    }


def _set_met_seed(seed: int) -> None:
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
