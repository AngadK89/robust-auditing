from __future__ import annotations

from typing import Any, Mapping


SUPPORTED_FORMATS = ("race", "arc")


def format_prompt(row: Mapping[str, Any], prompt_format: str) -> str:
    answers = f"(a){row['ans0']}(b){row['ans1']}(c){row['ans2']}"
    if prompt_format == "race":
        return f"{row['question']}\n{answers}\n{row['context']}"
    if prompt_format == "arc":
        return f"{row['context']}{row['question']}\n{answers}"
    raise ValueError(f"Unknown BBQ prompt format: {prompt_format}")
