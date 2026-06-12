#!/usr/bin/env python3
"""Summarise MT-Bench answer JSONL files for reproducibility checks.

This script intentionally does not score answers. It records lightweight sanity
statistics for generated answer files before GPT-4 judging or cached result
aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def question_ids(question_file: Path) -> set[int]:
    ids: set[int] = set()
    for row in load_jsonl(question_file):
        question_id = row.get("question_id")
        if not isinstance(question_id, int):
            raise ValueError(f"{question_file}: question_id must be an integer")
        ids.add(question_id)
    return ids


def summarise_answer_file(path: Path, expected_ids: set[int]) -> dict[str, Any]:
    rows = load_jsonl(path)
    observed_ids: set[int] = set()
    turn_lengths: list[int] = []
    empty_turns = 0
    total_choices = 0

    for row in rows:
        question_id = row.get("question_id")
        if isinstance(question_id, int):
            observed_ids.add(question_id)

        choices = row.get("choices", [])
        if not isinstance(choices, list):
            choices = []
        total_choices += len(choices)

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            turns = choice.get("turns", [])
            if not isinstance(turns, list):
                continue
            for turn in turns:
                text = "" if turn is None else str(turn)
                stripped = text.strip()
                if not stripped:
                    empty_turns += 1
                turn_lengths.append(len(stripped))

    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)

    return {
        "answer_file": str(path),
        "num_questions_expected": len(expected_ids),
        "num_answer_rows": len(rows),
        "num_question_ids_observed": len(observed_ids),
        "missing_question_ids": missing_ids,
        "extra_question_ids": extra_ids,
        "num_choices": total_choices,
        "num_turns": len(turn_lengths),
        "empty_turns": empty_turns,
        "min_turn_chars": min(turn_lengths) if turn_lengths else 0,
        "max_turn_chars": max(turn_lengths) if turn_lengths else 0,
        "mean_turn_chars": mean(turn_lengths) if turn_lengths else 0.0,
    }


def default_targets(answer_dir: Path) -> list[str]:
    return sorted(path.stem for path in answer_dir.glob("*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answer-dir",
        default="artifacts/mt_bench/model_answer",
        help="Directory containing MT-Bench model answer JSONL files.",
    )
    parser.add_argument(
        "--question-file",
        default="data/mt_bench/question.jsonl",
        help="MT-Bench question JSONL file used to check coverage.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        help="Model ids to summarise. Defaults to all JSONL files in --answer-dir.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to write summary JSON.",
    )
    args = parser.parse_args()

    answer_dir = Path(args.answer_dir)
    question_file = Path(args.question_file)
    output_file = Path(args.output_file)
    targets = args.targets if args.targets else default_targets(answer_dir)

    expected_ids = question_ids(question_file)
    summaries: dict[str, Any] = {}
    for target in targets:
        answer_file = answer_dir / f"{target}.jsonl"
        if not answer_file.exists():
            raise FileNotFoundError(f"Missing answer file for target {target}: {answer_file}")
        summaries[target] = summarise_answer_file(answer_file, expected_ids)

    payload = {
        "question_file": str(question_file),
        "answer_dir": str(answer_dir),
        "targets": summaries,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
