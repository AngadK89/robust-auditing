from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_INPUT_FIELDS = {
    "category",
    "example_id",
    "question_index",
    "context_condition",
    "question_polarity",
    "context",
    "question",
    "ans0",
    "ans1",
    "ans2",
    "label",
}
REQUIRED_METADATA_FIELDS = {"target_loc"}
REQUIRED_BBQ_FIELDS = REQUIRED_INPUT_FIELDS | REQUIRED_METADATA_FIELDS
ANSWER_KEYS = ("ans0", "ans1", "ans2")


def load_bbq_rows(upstream_root: Path, categories: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Load released BBQ JSONL rows and merge upstream additional metadata.

    The upstream R scorer filters out examples with missing target_loc. This
    loader does the same after preserving every released row field.
    """

    upstream_root = Path(upstream_root)
    metadata = _read_additional_metadata(upstream_root / "analysis_scripts" / "additional_metadata.csv")
    category_filter = set(categories) if categories is not None else None
    data_dir = upstream_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing BBQ data directory: {data_dir}")

    rows: list[dict[str, Any]] = []
    skipped_missing_target_loc = 0
    data_paths = sorted(data_dir.glob("*.jsonl"))
    if category_filter is not None:
        data_paths = [path for path in data_paths if path.stem in category_filter]
    for path in data_paths:
        for raw_row in _read_jsonl(path):
            _validate_input_row(raw_row, source=path)
            key = _metadata_key(raw_row)
            if key not in metadata:
                raise ValueError(f"Missing BBQ metadata for {key}")
            metadata_row = dict(metadata[key])
            if metadata_row.get("target_loc") is None:
                skipped_missing_target_loc += 1
                continue
            row = dict(raw_row)
            row.update(metadata_row)
            row["example_id"] = int(row["example_id"])
            row["question_index"] = str(row["question_index"])
            row["label"] = int(row["label"])
            row["target_loc"] = int(row["target_loc"])
            _add_answer_info_fields(row)
            _validate_scoring_row(row, source=path)
            rows.append(row)
    if not rows:
        raise ValueError(
            f"No BBQ rows with required target_loc found under {upstream_root}; "
            f"skipped_missing_target_loc={skipped_missing_target_loc}"
        )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _read_additional_metadata(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing BBQ metadata file: {path}")
    metadata: dict[tuple[str, int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {
                key: _clean_csv_value(value)
                for key, value in raw.items()
                if key is not None
            }
            if not {"category", "example_id", "question_index"} <= set(row):
                raise ValueError(f"Malformed BBQ metadata row in {path}: {raw}")
            if row.get("target_loc") is not None:
                row["target_loc"] = int(str(row["target_loc"]))
            row["example_id"] = int(str(row["example_id"]))
            row["question_index"] = str(row["question_index"])
            metadata[(str(row["category"]), row["example_id"], row["question_index"])] = row
    return metadata


def _clean_csv_value(value: str | None) -> Any:
    if value is None:
        return None
    if value in {"", "NA"}:
        return None
    return value


def _metadata_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["category"]), int(row["example_id"]), str(row["question_index"]))


def _validate_input_row(row: dict[str, Any], *, source: Path) -> None:
    missing = sorted(REQUIRED_INPUT_FIELDS - set(row))
    if missing:
        raise ValueError(f"Missing BBQ field(s) in {source}: {', '.join(missing)}")
    if "answer_info" not in row:
        raise ValueError(f"Missing BBQ answer_info in {source}")
    for answer_key in ANSWER_KEYS:
        if answer_key not in row["answer_info"]:
            raise ValueError(f"Missing answer_info.{answer_key} in {source}")


def _validate_scoring_row(row: dict[str, Any], *, source: Path) -> None:
    missing = sorted(REQUIRED_BBQ_FIELDS - set(row))
    if missing:
        raise ValueError(f"Missing BBQ scoring field(s) in {source}: {', '.join(missing)}")


def _add_answer_info_fields(row: dict[str, Any]) -> None:
    answer_info = row.get("answer_info", {})
    for answer_key in ANSWER_KEYS:
        values = answer_info.get(answer_key)
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"Malformed answer_info.{answer_key} for {row.get('category')}:{row.get('example_id')}")
        row[f"{answer_key}_text"] = values[0]
        row[f"{answer_key}_info"] = values[1]
