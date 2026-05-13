from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FairnessExample:
    text: str
    axis: str
    bucket: str
    descriptor: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_score_metadata(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "axis": self.axis,
            "bucket": self.bucket,
            "descriptor": self.descriptor,
            "metadata": dict(self.metadata),
        }


class DatasetSchemaError(ValueError):
    """Raised when an audit dataset does not expose the expected columns."""


class BaseAdapter:
    name: str
    dataset_id: str
    split: str = "train"
    data_files: list[str] | None = None
    required_columns: tuple[str, ...] = ()

    def validate_columns(self, dataset: Any) -> None:
        raw_columns = getattr(dataset, "columns", None)
        if raw_columns is None:
            raw_columns = getattr(dataset, "column_names", [])
        columns = set(raw_columns)
        missing = sorted(set(self.required_columns) - columns)
        if missing:
            joined = ", ".join(missing)
            raise DatasetSchemaError(f"{self.name} dataset is missing required columns: {joined}")

    def normalize(self, dataset: Any) -> Iterable[FairnessExample]:
        raise NotImplementedError

    def _iter_rows(self, dataset: Any) -> Iterable[tuple[int, Mapping[str, Any]]]:
        if hasattr(dataset, "iterrows"):
            for index, row in dataset.iterrows():
                yield int(index), row
            return

        for index, row in enumerate(dataset):
            yield index, row


class HolisticBiasAdapter(BaseAdapter):
    name = "HolisticBias"
    dataset_id = "fairnlp/holistic-bias"
    data_files = ["sentences.csv"]
    required_columns = ("text", "axis", "bucket", "descriptor")

    def normalize(self, dataset: Any) -> Iterable[FairnessExample]:
        self.validate_columns(dataset)
        for index, row in self._iter_rows(dataset):
            metadata = {"source_index": index}
            for column in ("template", "template_key", "template_id", "noun_phrase", "plural_noun_phrase"):
                if column in row and row[column] is not None:
                    metadata[column] = row[column]
            yield FairnessExample(
                text=str(row["text"]),
                axis=str(row["axis"]),
                bucket=str(row["bucket"]),
                descriptor=str(row["descriptor"]),
                metadata=metadata,
            )


class BoldAdapter(BaseAdapter):
    name = "BOLD"
    dataset_id = "AmazonScience/bold"
    required_columns = ("domain", "category", "name", "prompts")

    def normalize(self, dataset: Any) -> Iterable[FairnessExample]:
        self.validate_columns(dataset)
        for index, row in self._iter_rows(dataset):
            prompts = row["prompts"]
            if prompts is None:
                continue
            if isinstance(prompts, str):
                prompts = [prompts]
            for prompt_index, prompt in enumerate(prompts):
                yield FairnessExample(
                    text=str(prompt),
                    axis=str(row["domain"]),
                    bucket=str(row["category"]),
                    descriptor=str(row["category"]),
                    metadata={
                        "source_index": index,
                        "name": str(row["name"]),
                        "prompt_index": prompt_index,
                    },
                )
