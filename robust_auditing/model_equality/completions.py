from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from robust_auditing.model_equality.prompts import PromptRecord


@dataclass(frozen=True)
class CompletionRecord:
    suite: str
    prompt_id: str
    model_label: str
    sample_index: int
    prompt: str
    completion_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CompletionRecord":
        return cls(
            suite=str(payload["suite"]),
            prompt_id=str(payload["prompt_id"]),
            model_label=str(payload["model_label"]),
            sample_index=int(payload["sample_index"]),
            prompt=str(payload["prompt"]),
            completion_text=str(payload["completion_text"]),
            metadata=dict(payload.get("metadata", {})),
        )


def completion_records_to_met_arrays(
    records: Sequence[CompletionRecord],
    prompt_records: Sequence[PromptRecord],
    *,
    padding_length: int,
    pad_token_id: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    from model_equality_testing.utils import pad_to_length, tokenize_unicode

    if padding_length <= 0:
        raise ValueError("padding_length must be positive")
    prompt_to_index = {record.prompt_id: index for index, record in enumerate(prompt_records)}
    prompt_indices: list[int] = []
    texts: list[str] = []
    for record in records:
        if record.prompt_id not in prompt_to_index:
            raise ValueError(f"Completion references unknown prompt_id: {record.prompt_id}")
        prompt_indices.append(prompt_to_index[record.prompt_id])
        texts.append(record.completion_text)

    tokenized = tokenize_unicode(texts, pad_token_id=pad_token_id)
    tokenized = tokenized[:, :padding_length]
    padded = pad_to_length(tokenized, L=padding_length, pad_token_id=pad_token_id)
    return np.array(prompt_indices, dtype=np.int64), padded.astype(np.int64)


def completion_records_to_sample(
    records: Sequence[CompletionRecord],
    prompt_records: Sequence[PromptRecord],
    *,
    padding_length: int,
    pad_token_id: int = -1,
):
    from model_equality_testing.distribution import CompletionSample

    prompt_indices, completion_array = completion_records_to_met_arrays(
        records,
        prompt_records,
        padding_length=padding_length,
        pad_token_id=pad_token_id,
    )
    return CompletionSample(prompts=prompt_indices, completions=completion_array, m=len(prompt_records))


def completion_records_to_token_arrays(
    records: Sequence[CompletionRecord],
    prompt_records: Sequence[PromptRecord],
    *,
    padding_length: int,
    pad_token_id: int,
    eos_token_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if padding_length <= 0:
        raise ValueError("padding_length must be positive")
    prompt_to_index = {record.prompt_id: index for index, record in enumerate(prompt_records)}
    prompt_indices: list[int] = []
    rows: list[list[int]] = []
    for record in records:
        if record.prompt_id not in prompt_to_index:
            raise ValueError(f"Completion references unknown prompt_id: {record.prompt_id}")
        token_ids = record.metadata.get("completion_token_ids")
        if token_ids is None:
            raise ValueError("Token-space MET requires metadata['completion_token_ids']")
        prompt_indices.append(prompt_to_index[record.prompt_id])
        rows.append(
            normalize_completion_token_ids(
                token_ids,
                padding_length=padding_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        )
    return np.array(prompt_indices, dtype=np.int64), np.array(rows, dtype=np.int64)


def completion_records_to_token_sample(
    records: Sequence[CompletionRecord],
    prompt_records: Sequence[PromptRecord],
    *,
    padding_length: int,
    pad_token_id: int,
    eos_token_id: int | None = None,
):
    from model_equality_testing.distribution import CompletionSample

    prompt_indices, completion_array = completion_records_to_token_arrays(
        records,
        prompt_records,
        padding_length=padding_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    return CompletionSample(prompts=prompt_indices, completions=completion_array, m=len(prompt_records))


def normalize_completion_token_ids(
    token_ids: Sequence[int],
    *,
    padding_length: int,
    pad_token_id: int,
    eos_token_id: int | None = None,
) -> list[int]:
    normalized: list[int] = []
    for token_id in token_ids:
        token = int(token_id)
        normalized.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
        if len(normalized) >= padding_length:
            break
    if len(normalized) > padding_length:
        normalized = normalized[:padding_length]
    if len(normalized) < padding_length:
        normalized.extend([int(pad_token_id)] * (padding_length - len(normalized)))
    return normalized


def write_completion_records(path: Path, records: Iterable[CompletionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record.to_json(), ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def read_completion_records(path: Path) -> list[CompletionRecord]:
    return [
        CompletionRecord.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
