from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from robust_auditing.fairness.adapters import FairnessExample


NORMALIZED_PROMPTS = "normalized_prompts"
MODEL_RESPONSES = "model_responses"


def model_slug(model_id: str) -> str:
    last = model_id.split("/")[-1].lower()
    return (
        last.replace("olmo-2-0425-1b-instruct", "olmo2_1b_instruct")
        .replace("-", "_")
        .replace(".", "_")
    )


def metric_folder_name(metric: Any) -> str:
    cls = metric if isinstance(metric, type) else metric.__class__
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
    return name.removesuffix("_metric")


@dataclass(frozen=True)
class FairnessArtifactPaths:
    output_root: Path
    audit: str
    model_id: str

    @property
    def audit_dir(self) -> Path:
        return self.output_root / self.audit / model_slug(self.model_id)

    @property
    def normalized_prompts(self) -> Path:
        return self.audit_dir / "normalized_prompts.jsonl"

    @property
    def model_responses(self) -> Path:
        return self.audit_dir / "model_responses.jsonl"

    @property
    def metadata(self) -> Path:
        return self.audit_dir / "metadata.json"

    def artifact_path(self, artifact: str) -> Path:
        if artifact == NORMALIZED_PROMPTS:
            return self.normalized_prompts
        if artifact == MODEL_RESPONSES:
            return self.model_responses
        raise ValueError(f"Unknown fairness artifact: {artifact}")

    def metric_dir(self, metric: Any) -> Path:
        return self.audit_dir / "metrics" / metric_folder_name(metric)

    def metric_per_example(self, metric: Any) -> Path:
        return self.metric_dir(metric) / "per_example.jsonl"

    def metric_group_summary(self, metric: Any) -> Path:
        return self.metric_dir(metric) / "group_summary.csv"

    def metric_axis_summary(self, metric: Any) -> Path:
        return self.metric_dir(metric) / "axis_summary.csv"

    def metric_metadata(self, metric: Any) -> Path:
        return self.metric_dir(metric) / "metadata.json"


def fairness_example_to_record(example: FairnessExample) -> dict[str, Any]:
    return {
        "text": example.text,
        "axis": example.axis,
        "bucket": example.bucket,
        "descriptor": example.descriptor,
        "metadata": dict(example.metadata),
    }


def fairness_example_from_record(record: dict[str, Any]) -> FairnessExample:
    return FairnessExample(
        text=str(record["text"]),
        axis=str(record["axis"]),
        bucket=str(record["bucket"]),
        descriptor=str(record["descriptor"]),
        metadata=dict(record.get("metadata", {})),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_examples(path: Path) -> list[FairnessExample]:
    return [fairness_example_from_record(record) for record in read_jsonl(path)]
