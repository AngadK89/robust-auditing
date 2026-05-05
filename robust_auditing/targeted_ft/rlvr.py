from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RLVRCandidate:
    source_index: int | None
    prompt: str
    completion: str
    ground_truth: str
    is_correct: bool
    model_name: str
    sample_index: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + '\n')


def build_rlvr_dpo_pairs_from_candidates(candidates: Iterable[RLVRCandidate | Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        row = candidate.to_json() if isinstance(candidate, RLVRCandidate) else dict(candidate)
        key = (row.get('prompt_id', row.get('source_index')), str(row.get('prompt', '')), str(row.get('ground_truth', '')))
        grouped[key].append(row)

    pairs: list[dict[str, Any]] = []
    for (source_index, prompt, ground_truth), rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row.get('sample_index', row.get('candidate_index', 0))))
        correct = [row for row in rows if bool(row.get('is_correct'))]
        incorrect = [row for row in rows if not bool(row.get('is_correct'))]
        if not correct or not incorrect:
            continue
        chosen = correct[0]
        rejected = incorrect[0]
        pairs.append(
            {
                'prompt': prompt,
                'chosen': str(chosen.get('completion', '')),
                'rejected': str(rejected.get('completion', '')),
                'ground_truth': ground_truth,
                'source_index': chosen.get('source_index', source_index),
                **({'prompt_id': chosen.get('prompt_id')} if chosen.get('prompt_id') is not None else {}),
                'chosen_candidate_index': int(chosen.get('candidate_index', chosen.get('sample_index', 0))),
                'rejected_candidate_index': int(rejected.get('candidate_index', rejected.get('sample_index', 0))),
            }
        )
    return pairs
