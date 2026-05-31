from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_BBQ_ARTIFACT_ROOT = Path("artifacts/bbq")
DEFAULT_RUN_ID = "olmo2_instruct_clean_exact_chain_10k_seed0"


@dataclass(frozen=True)
class BBQSubsetPaths:
    artifact_root: Path
    subset_id: str

    @property
    def subset_dir(self) -> Path:
        return self.artifact_root / "subsets" / self.subset_id

    @property
    def examples(self) -> Path:
        return self.subset_dir / "examples.jsonl"

    @property
    def metadata(self) -> Path:
        return self.subset_dir / "metadata.json"


@dataclass(frozen=True)
class BBQRunPaths:
    artifact_root: Path = DEFAULT_BBQ_ARTIFACT_ROOT
    run_id: str = DEFAULT_RUN_ID

    @property
    def run_dir(self) -> Path:
        return self.artifact_root / "runs" / self.run_id

    @property
    def config(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def comparison_dir(self) -> Path:
        return self.run_dir / "comparison"

    @property
    def comparison_summary(self) -> Path:
        return self.comparison_dir / "summary.csv"

    @property
    def upstream_parity_dir(self) -> Path:
        return self.run_dir / "upstream_parity"

    @property
    def parity_report(self) -> Path:
        return self.upstream_parity_dir / "parity_report.json"

    def prediction_path(self, target_id: str, prompt_format: str) -> Path:
        return self.run_dir / "predictions" / target_id / prompt_format / "predictions.jsonl"

    def compat_path(self, target_id: str, category: str) -> Path:
        return self.run_dir / "predictions_compat" / target_id / f"{category}.jsonl"

    def metric_dir(self, target_id: str, prompt_format: str) -> Path:
        return self.run_dir / "metrics" / target_id / prompt_format


def write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


MappingLike = dict[str, Any]
