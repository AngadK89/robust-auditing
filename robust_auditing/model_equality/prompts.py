from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_WIKIPEDIA_DATASET_ID = "Cohere/wikipedia-2023-11-embed-multilingual-v3"
DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE = "{language}"
DEFAULT_WIKIPEDIA_LANGUAGES = ("en", "de", "fr", "es", "ru")
DEFAULT_ULTRACHAT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DEFAULT_ULTRACHAT_SPLIT = "test_gen"
DEFAULT_HUMANEVAL_DATASET_ID = "openai/openai_humaneval"
DEFAULT_HUMANEVAL_SPLIT = "test"
PROMPT_SUITES = ("wikipedia", "ultrachat", "humaneval")


@dataclass(frozen=True)
class PromptRecord:
    suite: str
    prompt_id: str
    text: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PromptRecord":
        return cls(
            suite=str(payload["suite"]),
            prompt_id=str(payload["prompt_id"]),
            text=str(payload["text"]),
            metadata=dict(payload.get("metadata", {})),
        )


def normalize_wikipedia_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    language: str,
    prompt_chars: int = 100,
) -> PromptRecord | None:
    text = _collapse_whitespace(str(row.get("text", "")))
    if not text:
        return None
    snippet = text[:prompt_chars]
    source_id = str(row.get("id") or row.get("page_id") or row_index)
    metadata: dict[str, Any] = {
        "row_index": row_index,
        "language": language,
        "source_id": source_id,
    }
    if row.get("title") is not None:
        metadata["title"] = str(row["title"])
    return PromptRecord(
        suite="wikipedia",
        prompt_id=f"wikipedia:{_safe_id(language)}:{_safe_id(source_id)}",
        text=(
            "Continue the paragraph. Do not output anything except the continuation to the paragraph. "
            f'Start the continuation immediately.\n"{snippet}..."'
        ),
        metadata=metadata,
    )


def normalize_ultrachat_row(row: Mapping[str, Any], *, row_index: int) -> PromptRecord | None:
    prompt = str(row["prompt"]) if row.get("prompt") is not None else _first_user_message(row)
    if prompt is None:
        return None
    prompt = _collapse_whitespace(prompt)
    if not prompt:
        return None
    source_id = row.get("id") or row.get("prompt_id") or row_index
    return PromptRecord(
        suite="ultrachat",
        prompt_id=f"ultrachat:{_safe_id(source_id)}",
        text=prompt,
        metadata={"row_index": row_index},
    )


def normalize_humaneval_row(row: Mapping[str, Any], *, row_index: int) -> PromptRecord | None:
    prompt = str(row.get("prompt", ""))
    if not prompt.strip():
        return None
    task_id = str(row.get("task_id") or row_index)
    return PromptRecord(
        suite="humaneval",
        prompt_id=f"humaneval:{_safe_id(task_id)}",
        text=(
            "Complete the code. Do not output anything except the completion. "
            "Start the continuation immediately.\n```\n"
            + prompt
        ),
        metadata={"row_index": row_index, "task_id": task_id},
    )


def load_prompt_suite(
    suite: str,
    *,
    max_prompts: int,
    seed: int = 0,
    wikipedia_languages: Sequence[str] = DEFAULT_WIKIPEDIA_LANGUAGES,
    wikipedia_dataset_id: str = DEFAULT_WIKIPEDIA_DATASET_ID,
    wikipedia_config_template: str = DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE,
    ultrachat_dataset_id: str = DEFAULT_ULTRACHAT_DATASET_ID,
    ultrachat_split: str = DEFAULT_ULTRACHAT_SPLIT,
    humaneval_dataset_id: str = DEFAULT_HUMANEVAL_DATASET_ID,
    humaneval_split: str = DEFAULT_HUMANEVAL_SPLIT,
) -> list[PromptRecord]:
    if max_prompts <= 0:
        raise ValueError("max_prompts must be positive")
    if suite == "wikipedia":
        return load_wikipedia_prompts(
            max_prompts=max_prompts,
            languages=wikipedia_languages,
            dataset_id=wikipedia_dataset_id,
            config_template=wikipedia_config_template,
        )
    if suite == "ultrachat":
        return load_ultrachat_prompts(
            max_prompts=max_prompts,
            dataset_id=ultrachat_dataset_id,
            split=ultrachat_split,
            seed=seed,
        )
    if suite == "humaneval":
        return load_humaneval_prompts(max_prompts=max_prompts, dataset_id=humaneval_dataset_id, split=humaneval_split)
    raise ValueError(f"Unknown prompt suite: {suite}")


def load_wikipedia_prompts(
    *,
    max_prompts: int,
    languages: Sequence[str] = DEFAULT_WIKIPEDIA_LANGUAGES,
    dataset_id: str = DEFAULT_WIKIPEDIA_DATASET_ID,
    config_template: str = DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE,
) -> list[PromptRecord]:
    from datasets import load_dataset

    prompts: list[PromptRecord] = []
    per_language = max(1, math.ceil(max_prompts / len(languages)))
    for language in languages:
        config_name = config_template.format(language=language)
        dataset = load_dataset(dataset_id, config_name, split="train", streaming=True)
        remove_columns = [
            column
            for column in ("url", "_id", "title", "emb")
            if column in getattr(dataset, "column_names", [])
        ]
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)
        added = 0
        for row_index, row in enumerate(dataset):
            record = normalize_wikipedia_row(row, row_index=row_index, language=language)
            if record is None:
                continue
            prompts.append(record)
            added += 1
            if added >= per_language or len(prompts) >= max_prompts:
                break
        if len(prompts) >= max_prompts:
            break
    return prompts[:max_prompts]


def load_ultrachat_prompts(
    *,
    max_prompts: int,
    dataset_id: str = DEFAULT_ULTRACHAT_DATASET_ID,
    split: str = DEFAULT_ULTRACHAT_SPLIT,
    seed: int = 0,
) -> list[PromptRecord]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, streaming=True)
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=max(1000, max_prompts * 20))
    records: list[PromptRecord] = []
    for row_index, row in enumerate(dataset):
        record = normalize_ultrachat_row(row, row_index=row_index)
        if record is None:
            continue
        records.append(record)
        if len(records) >= max_prompts:
            break
    return records


def load_humaneval_prompts(
    *,
    max_prompts: int,
    dataset_id: str = DEFAULT_HUMANEVAL_DATASET_ID,
    split: str = DEFAULT_HUMANEVAL_SPLIT,
) -> list[PromptRecord]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    records: list[PromptRecord] = []
    for row_index, row in enumerate(dataset):
        record = normalize_humaneval_row(row, row_index=row_index)
        if record is None:
            continue
        records.append(record)
        if len(records) >= max_prompts:
            break
    return records


def write_prompt_records(path: Path, records: Iterable[PromptRecord]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record.to_json(), ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def _first_user_message(row: Mapping[str, Any]) -> str | None:
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, str):
        return messages
    if isinstance(messages, Sequence):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            if role not in {"user", "human"}:
                continue
            content = message.get("content", message.get("value"))
            return str(content) if content is not None else None
    for key in ("prompt", "instruction", "text"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_id(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("_") or "unknown"
