from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ANSWER_LETTERS = ("A", "B", "C", "D")
OPTION_COLUMNS = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}

SYSTEM_PROMPT = (
    "You are answering a medical multiple-choice question. "
    "Answer with exactly one letter: A, B, C, or D. "
    "Do not include explanation text."
)


@dataclass(frozen=True)
class MedMCQAExample:
    example_id: str
    question: str
    options: dict[str, str]
    answer: str
    choice_type: str
    subject_name: str | None
    topic_name: str | None
    source_index: int

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "question": self.question,
            "options": dict(self.options),
            "answer": self.answer,
            "choice_type": self.choice_type,
            "subject_name": self.subject_name,
            "topic_name": self.topic_name,
            "source_index": self.source_index,
        }


def answer_letter(cop: Any) -> str:
    try:
        index = int(cop)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MedMCQA answer index: {cop!r}") from exc
    if index < 0 or index >= len(ANSWER_LETTERS):
        raise ValueError(f"Invalid MedMCQA answer index: {cop!r}")
    return ANSWER_LETTERS[index]


def sample_rows(rows: Sequence[Mapping[str, Any]], max_examples: int | None, seed: int) -> list[Mapping[str, Any]]:
    materialized = list(rows)
    if max_examples is None or max_examples >= len(materialized):
        return materialized
    rng = random.Random(seed)
    indices = rng.sample(range(len(materialized)), max_examples)
    return [materialized[index] for index in indices]


def normalize_row(row: Mapping[str, Any], source_index: int) -> MedMCQAExample:
    options = {letter: _clean_text(row[column]) for letter, column in OPTION_COLUMNS.items()}
    return MedMCQAExample(
        example_id=str(row.get("id", source_index)),
        question=_clean_text(row["question"]),
        options=options,
        answer=answer_letter(row["cop"]),
        choice_type=str(row.get("choice_type") or "unknown"),
        subject_name=_optional_text(row.get("subject_name")),
        topic_name=_optional_text(row.get("topic_name")),
        source_index=source_index,
    )


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> list[MedMCQAExample]:
    examples: list[MedMCQAExample] = []
    for source_index, row in enumerate(rows):
        try:
            examples.append(normalize_row(row, source_index=source_index))
        except ValueError:
            continue
    return examples


def format_prompt(example: MedMCQAExample) -> str:
    options = "\n".join(f"{letter}. {example.options[letter]}" for letter in ANSWER_LETTERS)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {example.question}\n"
        f"{options}\n\n"
        "Answer:"
    )


def to_grpo_record(example: MedMCQAExample) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": format_prompt(example)}],
        "answer": example.answer,
        "example_id": example.example_id,
        "choice_type": example.choice_type,
        "subject_name": example.subject_name,
    }


def _clean_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = _clean_text(value)
    return text or None
