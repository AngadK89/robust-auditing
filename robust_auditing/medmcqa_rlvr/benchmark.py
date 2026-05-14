from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import torch
from tqdm.auto import tqdm

from robust_auditing.medmcqa_rlvr.data import ANSWER_LETTERS, MedMCQAExample, format_prompt
from robust_auditing.medmcqa_rlvr.rewards import extract_answer


@dataclass(frozen=True)
class ForcedChoiceResult:
    example_id: str
    predicted: str
    answer: str
    correct: bool
    scores: dict[str, float]
    choice_type: str
    subject_name: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "predicted": self.predicted,
            "answer": self.answer,
            "correct": self.correct,
            "scores": dict(self.scores),
            "choice_type": self.choice_type,
            "subject_name": self.subject_name,
        }


@dataclass(frozen=True)
class GeneratedAnswerResult:
    example_id: str
    generated_text: str
    predicted: str | None
    answer: str
    correct: bool
    choice_type: str
    subject_name: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "generated_text": self.generated_text,
            "predicted": self.predicted,
            "answer": self.answer,
            "correct": self.correct,
            "choice_type": self.choice_type,
            "subject_name": self.subject_name,
        }


def choose_forced_choice(
    example: MedMCQAExample,
    score_candidate: Callable[[MedMCQAExample, str], float],
) -> ForcedChoiceResult:
    scores = {candidate: float(score_candidate(example, candidate)) for candidate in ANSWER_LETTERS}
    predicted = max(scores, key=scores.get)
    return ForcedChoiceResult(
        example_id=example.example_id,
        predicted=predicted,
        answer=example.answer,
        correct=predicted == example.answer,
        scores=scores,
        choice_type=example.choice_type,
        subject_name=example.subject_name,
    )


def evaluate_forced_choice(
    examples: Iterable[MedMCQAExample],
    model: Any,
    tokenizer: Any,
    batch_size: int = 8,
) -> list[ForcedChoiceResult]:
    model.eval()
    results: list[ForcedChoiceResult] = []
    materialized = list(examples)
    for start in tqdm(range(0, len(materialized), batch_size), desc="Forced-choice eval", unit="batch"):
        batch = materialized[start : start + batch_size]
        batch_scores = answer_token_logits_batch(batch, model, tokenizer)
        for example in batch:
            scores = batch_scores[example.example_id]
            predicted = max(scores, key=scores.get)
            results.append(
                ForcedChoiceResult(
                    example_id=example.example_id,
                    predicted=predicted,
                    answer=example.answer,
                    correct=predicted == example.answer,
                    scores=scores,
                    choice_type=example.choice_type,
                    subject_name=example.subject_name,
                )
            )
    return results


def candidate_logprob(example: MedMCQAExample, candidate: str, model: Any, tokenizer: Any) -> float:
    return answer_token_logits_batch([example], model, tokenizer)[example.example_id][candidate]


def candidate_logprobs_batch(
    examples: list[MedMCQAExample],
    model: Any,
    tokenizer: Any,
) -> dict[str, dict[str, float]]:
    return answer_token_logits_batch(examples, model, tokenizer)


def answer_token_logits_batch(
    examples: list[MedMCQAExample],
    model: Any,
    tokenizer: Any,
) -> dict[str, dict[str, float]]:
    prompts = [render_prompt(example, tokenizer) for example in examples]
    token_ids = answer_token_ids(tokenizer)
    device = next(model.parameters()).device
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    last_idx = attention_mask.sum(dim=-1) - 1
    b_idx = torch.arange(input_ids.size(0), device=device)
    answer_logits = logits[b_idx, last_idx, :][:, token_ids]

    results: dict[str, dict[str, float]] = {}
    for example, row in zip(examples, answer_logits):
        results[example.example_id] = {
            letter: float(score.item())
            for letter, score in zip(ANSWER_LETTERS, row)
        }
    return results


def _candidate_logprob_unbatched(example: MedMCQAExample, candidate: str, model: Any, tokenizer: Any) -> float:
    return answer_token_logits_batch([example], model, tokenizer)[example.example_id][candidate]


def answer_token_ids(tokenizer: Any) -> list[int]:
    token_ids: list[int] = []
    for letter in ANSWER_LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer letter {letter!r} did not tokenize to one token: {ids}")
        token_ids.append(ids[0])
    return token_ids


def evaluate_generation(
    examples: Iterable[MedMCQAExample],
    model: Any,
    tokenizer: Any,
    batch_size: int = 8,
    max_new_tokens: int = 8,
) -> list[GeneratedAnswerResult]:
    rows: list[GeneratedAnswerResult] = []
    materialized = list(examples)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    for start in tqdm(range(0, len(materialized), batch_size), desc="Generation eval", unit="batch"):
        batch = materialized[start : start + batch_size]
        prompts = [render_prompt(example, tokenizer) for example in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        prompt_width = encoded["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for example, output_ids in zip(batch, generated):
            response_ids = output_ids[prompt_width:]
            text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            predicted = extract_answer(text)
            rows.append(
                GeneratedAnswerResult(
                    example_id=example.example_id,
                    generated_text=text,
                    predicted=predicted,
                    answer=example.answer,
                    correct=predicted == example.answer,
                    choice_type=example.choice_type,
                    subject_name=example.subject_name,
                )
            )
    return rows


def render_prompt(example: MedMCQAExample, tokenizer: Any) -> str:
    prompt = format_prompt(example)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def summarize_forced_choice(results: list[ForcedChoiceResult]) -> dict[str, Any]:
    frame = pd.DataFrame([result.to_record() for result in results])
    return _summarize_frame(frame, prefix="forced_choice")


def summarize_generation(results: list[GeneratedAnswerResult]) -> dict[str, Any]:
    frame = pd.DataFrame([result.to_record() for result in results])
    summary = _summarize_frame(frame, prefix="generated")
    if len(frame) == 0:
        summary["generated_parse_rate"] = 0.0
    else:
        summary["generated_parse_rate"] = float(frame["predicted"].notna().mean())
        summary["generated_invalid_rate"] = 1.0 - summary["generated_parse_rate"]
    return summary


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _summarize_frame(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if len(frame) == 0:
        return {f"{prefix}_accuracy": 0.0, "count": 0}
    summary: dict[str, Any] = {
        f"{prefix}_accuracy": float(frame["correct"].mean()),
        "count": int(len(frame)),
    }
    summary[f"{prefix}_accuracy_by_choice_type"] = (
        frame.groupby("choice_type")["correct"].mean().sort_index().to_dict()
    )
    subject_frame = frame.dropna(subset=["subject_name"])
    summary[f"{prefix}_accuracy_by_subject"] = (
        subject_frame.groupby("subject_name")["correct"].mean().sort_index().to_dict()
        if len(subject_frame)
        else {}
    )
    return summary
