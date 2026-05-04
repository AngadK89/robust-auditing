from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any


_METADATA_EXCLUDE_KEYS = {
    "candidate_completions",
    "candidates",
    "chosen",
    "completion",
    "completions",
    "dataset_class",
    "dataset_name",
    "ground_truth",
    "messages",
    "objective",
    "prompt",
    "rejected",
    "reward_model",
    "score",
    "scores",
    "source_index",
    "text",
    "verifier_scores",
}


def build_dpo_pairs(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        if "chosen" not in row or "rejected" not in row:
            continue
        pairs.append(
            {
                "dataset_name": row.get("dataset_name"),
                "objective": row.get("objective", "dpo"),
                "prompt": row.get("prompt"),
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "metadata": _metadata(row),
            }
        )
    return pairs


def build_sft_examples(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if "messages" in row:
            text, assistant_spans = _render_messages(row["messages"])
            tokens, offsets = _tokenize_with_offsets(text)
            assistant_token_mask = [
                any(_overlaps(offset, span) for span in assistant_spans) for offset in offsets
            ]
            example: dict[str, Any] = {
                "text": text,
                "messages": row["messages"],
                "tokens": tokens,
                "assistant_token_mask": assistant_token_mask,
                "assistant_char_spans": assistant_spans,
                "metadata": _metadata(row),
            }
        elif "text" in row:
            text = str(row["text"])
            tokens, _ = _tokenize_with_offsets(text)
            example = {
                "text": text,
                "tokens": tokens,
                "assistant_token_mask": [True] * len(tokens),
                "assistant_char_spans": [(0, len(text))] if text else [],
                "metadata": _metadata(row),
            }
        else:
            continue
        examples.append(example)
    return examples


def build_nll_anchor_examples(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row.get("dataset_name") != "holistic_bias" or "text" not in row:
            continue
        examples.append(
            {
                "dataset_name": "holistic_bias",
                "objective": "nll_anchor",
                "text": str(row["text"]),
                "metadata": _metadata(row),
            }
        )
    return examples


def build_rlvr_math_records(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    verifier_dpo: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    for row in rows:
        scored = _scored_candidates(row)
        if len(scored) >= 2:
            scored = sorted(scored, key=lambda item: item[1], reverse=True)
            chosen, chosen_score = scored[0]
            rejected, rejected_score = scored[-1]
            verifier_dpo.append(
                {
                    "dataset_name": "rlvr_math",
                    "objective": "dpo",
                    "prompt": row.get("messages", row.get("prompt")),
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "ground_truth": row.get("ground_truth"),
                    },
                }
            )
            continue

        eval_records.append(
            {
                "dataset_name": "rlvr_math",
                "messages": row.get("messages"),
                "ground_truth": row.get("ground_truth"),
                "metadata": _metadata(row),
            }
        )
    return {"verifier_dpo": verifier_dpo, "eval": eval_records}


def _render_messages(messages: Iterable[Mapping[str, Any]]) -> tuple[str, list[tuple[int, int]]]:
    chunks: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        prefix = f"<|{role}|>\n"
        chunks.append(prefix)
        cursor += len(prefix)
        start = cursor
        chunks.append(content)
        cursor += len(content)
        if role == "assistant" and content:
            assistant_spans.append((start, cursor))
        chunks.append("\n")
        cursor += 1
    return "".join(chunks), assistant_spans


def _tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append(match.span())
    return tokens, offsets


def _overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _scored_candidates(row: Mapping[str, Any]) -> list[tuple[Any, float]]:
    raw_candidates = row.get("candidate_completions", row.get("candidates", row.get("completions")))
    if raw_candidates is None:
        return []

    if isinstance(raw_candidates, Mapping):
        raw_candidates = raw_candidates.values()

    scores = row.get("scores", row.get("verifier_scores"))
    scored: list[tuple[Any, float]] = []
    for index, candidate in enumerate(raw_candidates):
        if isinstance(candidate, Mapping):
            text = candidate.get("completion", candidate.get("text", candidate.get("answer")))
            score = candidate.get("score", candidate.get("verifier_score"))
        else:
            text = candidate
            score = scores[index] if scores is not None and index < len(scores) else None
        if text is not None and score is not None:
            scored.append((text, float(score)))
    return scored


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in _METADATA_EXCLUDE_KEYS}
