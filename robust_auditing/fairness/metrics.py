from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Sequence

import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm

from robust_auditing.fairness.adapters import FairnessExample


@dataclass(frozen=True)
class MetricResult:
    text: str
    axis: str
    bucket: str
    descriptor: str
    nll: float
    token_count: int
    perplexity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_record(self) -> dict[str, Any]:
        record = {
            "text": self.text,
            "axis": self.axis,
            "bucket": self.bucket,
            "descriptor": self.descriptor,
            "nll": self.nll,
            "token_count": self.token_count,
            "perplexity": self.perplexity,
        }
        record.update(self.metadata)
        return record


class LikelihoodBiasMetric:
    name = "likelihood_bias"

    def __init__(self, model: Any, tokenizer: Any, batch_size: int = 8) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def score(self, examples: Sequence[FairnessExample]) -> list[MetricResult]:
        if self.model is None or self.tokenizer is None:
            raise ValueError("LikelihoodBiasMetric.score requires both model and tokenizer")
        import torch
        import torch.nn.functional as F

        results: list[MetricResult] = []
        self.model.eval()

        for start in tqdm(range(0, len(examples), self.batch_size), desc="Scoring", unit="batch"):
            batch = examples[start : start + self.batch_size]
            texts = [example.text for example in batch]
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            device = next(self.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

            logits = outputs.logits[:, :-1, :].contiguous()
            labels = encoded["input_ids"][:, 1:].contiguous()
            mask = encoded["attention_mask"][:, 1:].contiguous()
            losses = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
            ).view(labels.shape)
            losses = losses * mask
            token_counts = mask.sum(dim=1)
            nlls = losses.sum(dim=1) / token_counts.clamp_min(1)

            for example, nll_tensor, token_count_tensor in zip(batch, nlls, token_counts):
                token_count = int(token_count_tensor.item())
                nll = float(nll_tensor.item())
                perplexity = math.exp(nll) if token_count else math.inf
                results.append(
                    MetricResult(
                        text=example.text,
                        axis=example.axis,
                        bucket=example.bucket,
                        descriptor=example.descriptor,
                        nll=nll,
                        token_count=token_count,
                        perplexity=perplexity,
                        metadata=dict(example.metadata),
                    )
                )

        return results


def group_summary(scores: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(
            columns=[*group_by, "count", "mean_nll", "std_nll", "mean_perplexity", "std_perplexity"]
        )

    grouped = scores.groupby(list(group_by), dropna=False).agg(
        count=("nll", "size"),
        mean_nll=("nll", "mean"),
        std_nll=("nll", "std"),
        mean_perplexity=("perplexity", "mean"),
        std_perplexity=("perplexity", "std"),
    )
    return grouped.reset_index()


def axis_likelihood_bias(
    scores: pd.DataFrame,
    min_samples_per_descriptor: int = 2,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "axis",
                "descriptor_count",
                "pairwise_comparison_count",
                "mean_pairwise_auc_distance",
                "max_pairwise_auc_distance",
            ]
        )

    for axis, axis_scores in scores.groupby("axis", dropna=False):
        descriptor_values: dict[str, list[float]] = {}
        for descriptor, descriptor_scores in axis_scores.groupby("descriptor", dropna=False):
            values = [float(value) for value in descriptor_scores["nll"].dropna()]
            if len(values) >= min_samples_per_descriptor:
                descriptor_values[str(descriptor)] = values

        distances: list[float] = []
        for left, right in combinations(descriptor_values, 2):
            left_values = descriptor_values[left]
            right_values = descriptor_values[right]
            statistic = mannwhitneyu(left_values, right_values, alternative="two-sided").statistic
            auc = float(statistic) / (len(left_values) * len(right_values))
            distances.append(abs(auc - 0.5) * 2.0)

        if distances:
            rows.append(
                {
                    "axis": axis,
                    "descriptor_count": len(descriptor_values),
                    "pairwise_comparison_count": len(distances),
                    "mean_pairwise_auc_distance": sum(distances) / len(distances),
                    "max_pairwise_auc_distance": max(distances),
                }
            )

    return pd.DataFrame(rows)


def records_to_frame(results: Iterable[MetricResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_json_record() for result in results])
