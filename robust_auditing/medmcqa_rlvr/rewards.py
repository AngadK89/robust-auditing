from __future__ import annotations

import re
from typing import Any


ANSWER_TAG_RE = re.compile(r"^\s*<answer>\s*([ABCDabcd])\s*</answer>\s*$")
BARE_ANSWER_RE = re.compile(r"^\s*([ABCDabcd])\s*$")
ANSWER_SENTENCE_RE = re.compile(r"\banswer\s+is\s+([ABCDabcd])\b", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    tag_match = ANSWER_TAG_RE.match(text)
    if tag_match:
        return tag_match.group(1).upper()

    bare_match = BARE_ANSWER_RE.match(text)
    if bare_match:
        return bare_match.group(1).upper()

    sentence_matches = ANSWER_SENTENCE_RE.findall(text)
    if len(sentence_matches) == 1:
        return sentence_matches[0].upper()
    return None


def correctness_reward(completions: list[list[dict[str, str]]], answer: list[str], **_: Any) -> list[float]:
    responses = _completion_texts(completions)
    return [2.0 if extract_answer(response) == expected else 0.0 for response, expected in zip(responses, answer)]


def format_reward(completions: list[list[dict[str, str]]], **_: Any) -> list[float]:
    rewards: list[float] = []
    for response in _completion_texts(completions):
        if BARE_ANSWER_RE.match(response):
            rewards.append(0.5)
        elif ANSWER_TAG_RE.match(response):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def invalid_answer_penalty(completions: list[list[dict[str, str]]], **_: Any) -> list[float]:
    return [0.0 if extract_answer(response) is not None else -0.5 for response in _completion_texts(completions)]


def _completion_texts(completions: list[list[dict[str, str]]]) -> list[str]:
    return [completion[0].get("content", "") for completion in completions]
