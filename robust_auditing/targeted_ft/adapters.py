from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class TargetedFTSchemaError(ValueError):
    """Raised when a targeted fine-tuning source lacks required fields."""


class TargetedFTAdapter:
    name: str
    dataset_name: str
    dataset_id: str
    dataset_class: str
    objective: str
    objective_plugin: str | None = None
    reward_model: str | None = None
    dataset_config: str | None = None
    split: str = "train"
    data_files: list[str] | None = None
    required_columns: tuple[str, ...] = ()

    def validate_columns(self, dataset: Any) -> None:
        columns = set(self._columns(dataset))
        missing = sorted(set(self.required_columns) - columns)
        if missing:
            joined = ", ".join(missing)
            raise TargetedFTSchemaError(f"{self.name} dataset is missing required columns: {joined}")

    def validate_row(self, row: Mapping[str, Any]) -> None:
        missing = sorted(set(self.required_columns) - set(row))
        if missing:
            joined = ", ".join(missing)
            raise TargetedFTSchemaError(f"{self.name} dataset row is missing required columns: {joined}")

    def normalize(self, dataset: Any) -> Iterable[dict[str, Any]]:
        if self._has_column_metadata(dataset):
            self.validate_columns(dataset)
        for source_index, row in self._iter_rows(dataset):
            self.validate_row(row)
            yield self._normalize_row(source_index, row)

    def _normalize_row(self, source_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _base_record(self, source_index: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "dataset_class": self.dataset_class,
            "dataset_name": self.dataset_name,
            "objective": self.objective,
            "source_index": source_index,
        }
        if self.objective_plugin is not None:
            record["objective_plugin"] = self.objective_plugin
        if self.reward_model is not None:
            record["reward_model"] = self.reward_model
        return record

    def _columns(self, dataset: Any) -> list[str]:
        raw_columns = getattr(dataset, "columns", None)
        if raw_columns is not None:
            return list(raw_columns)
        raw_columns = getattr(dataset, "column_names", None)
        if raw_columns is not None:
            return list(raw_columns)
        return []

    def _has_column_metadata(self, dataset: Any) -> bool:
        return getattr(dataset, "columns", None) is not None or getattr(dataset, "column_names", None) is not None

    def _iter_rows(self, dataset: Any) -> Iterable[tuple[int, Mapping[str, Any]]]:
        if hasattr(dataset, "iterrows"):
            for index, row in dataset.iterrows():
                yield int(index), row
            return

        for index, row in enumerate(dataset):
            if not isinstance(row, Mapping):
                raise TypeError(f"{self.name} rows must be mapping-like")
            yield index, row


class HolisticBiasTargetedAdapter(TargetedFTAdapter):
    name = "HolisticBias"
    dataset_name = "holistic_bias"
    dataset_id = "fairnlp/holistic-bias"
    dataset_class = "fairness_audit"
    objective = "holistic_bias_anchor"
    objective_plugin = "nll_anchor"
    dataset_config = "sentences"
    split = "test"
    data_files = ["sentences.csv"]
    required_columns = ("text", "axis", "bucket", "descriptor")

    def _normalize_row(self, source_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        record = self._base_record(source_index)
        record.update(
            {
                "text": str(row["text"]),
                "axis": str(row["axis"]),
                "bucket": str(row["bucket"]),
                "descriptor": str(row["descriptor"]),
            }
        )
        return record


class Tulu3SFTAdapter(TargetedFTAdapter):
    name = "Tulu 3 SFT"
    dataset_name = "tulu3_sft"
    dataset_id = "allenai/tulu-3-sft-olmo-2-mixture-0225"
    dataset_class = "non_fairness_audit"
    objective = "sft"
    required_columns = ("id", "messages", "source")

    def _normalize_row(self, source_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        record = self._base_record(source_index)
        record.update({"id": row["id"], "messages": row["messages"], "source": row["source"]})
        return record


class PreferenceMixDPOAdapter(TargetedFTAdapter):
    name = "OLMo preference mix"
    dataset_name = "preference_mix"
    dataset_id = "allenai/olmo-2-0425-1b-preference-mix"
    dataset_class = "non_fairness_audit"
    objective = "dpo"
    required_columns = ("chosen", "rejected", "chosen_model", "rejected_model", "id", "source")

    def _normalize_row(self, source_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        record = self._base_record(source_index)
        record.update(
            {
                "id": row["id"],
                "source": row["source"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "chosen_model": row["chosen_model"],
                "rejected_model": row["rejected_model"],
            }
        )
        return record


class RLVRMathRewardAdapter(TargetedFTAdapter):
    name = "RLVR-MATH"
    dataset_name = "rlvr_math"
    dataset_id = "allenai/RLVR-MATH"
    dataset_class = "non_fairness_audit"
    objective = "rl_reward"
    reward_model = "math_verifier_score"
    required_columns = ("messages", "ground_truth", "dataset", "constraint_type", "constraint")

    def _normalize_row(self, source_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        record = self._base_record(source_index)
        record.update(
            {
                "messages": row["messages"],
                "ground_truth": row["ground_truth"],
                "dataset": row["dataset"],
                "constraint_type": row["constraint_type"],
                "constraint": row["constraint"],
            }
        )
        if "verifier_candidates" in row:
            record["verifier_candidates"] = row["verifier_candidates"]
        return record


class HHRLHFInvertedDPOAdapter(TargetedFTAdapter):
    name = "HH-RLHF"
    dataset_name = "hh_rlhf"
    dataset_id = "Anthropic/hh-rlhf"
    dataset_class = "off_audit"
    objective = "inverted_dpo"
    required_columns = ("chosen", "rejected")
    excluded_sources = {"red-team-attempts"}

    def normalize(self, dataset: Any) -> Iterable[dict[str, Any]]:
        if self._has_column_metadata(dataset):
            self.validate_columns(dataset)
        for source_index, row in self._iter_rows(dataset):
            self.validate_row(row)
            if str(row.get("source", "")) in self.excluded_sources:
                continue
            yield self._normalize_row(source_index, row)

    def _normalize_row(self, source_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        record = self._base_record(source_index)
        record.update(
            {
                "chosen": row["rejected"],
                "rejected": row["chosen"],
            }
        )
        if "source" in row:
            record["source"] = row["source"]
        return record


ADAPTERS: dict[str, TargetedFTAdapter] = {
    "holistic_bias": HolisticBiasTargetedAdapter(),
    "tulu3_sft": Tulu3SFTAdapter(),
    "preference_mix": PreferenceMixDPOAdapter(),
    "rlvr_math": RLVRMathRewardAdapter(),
    "hh_rlhf": HHRLHFInvertedDPOAdapter(),
}
