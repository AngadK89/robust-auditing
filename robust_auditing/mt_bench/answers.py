from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .fastchat_olmo2 import register_olmo2_fastchat_support
from .targets import MTBenchTarget


def normalize_torch_dtype(dtype: str | None):
    if dtype is None:
        return None
    import torch

    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    normalized = dtype.lower()
    return aliases.get(normalized, dtype)


def write_fake_model_answers(
    *,
    question_file: Path,
    answer_file: Path,
    model_id: str,
    answer_text: str,
) -> None:
    questions = [json.loads(line) for line in question_file.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for question in sorted(questions, key=lambda row: row["question_id"]):
        rows.append(
            {
                "question_id": question["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": [
                    {
                        "index": 0,
                        "turns": [answer_text for _ in question["turns"]],
                    }
                ],
                "tstamp": time.time(),
            }
        )

    answer_file.parent.mkdir(parents=True, exist_ok=True)
    answer_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def generate_model_answer(
    *,
    target: MTBenchTarget,
    question_file: Path,
    answer_dir: Path,
    max_new_token: int,
    num_choices: int,
    num_gpus_per_model: int,
    num_gpus_total: int,
    max_gpu_memory: str | None,
    dtype: str | None,
    question_begin: int | None = None,
    question_end: int | None = None,
) -> Path:
    from fastchat.llm_judge.gen_model_answer import reorg_answer_file, run_eval

    register_olmo2_fastchat_support()
    answer_file = answer_dir / f"{target.model_id}.jsonl"
    if answer_file.exists():
        answer_file.unlink()
    run_eval(
        model_path=target.model_path,
        model_id=target.model_id,
        question_file=str(question_file),
        question_begin=question_begin,
        question_end=question_end,
        answer_file=str(answer_file),
        max_new_token=max_new_token,
        num_choices=num_choices,
        num_gpus_per_model=num_gpus_per_model,
        num_gpus_total=num_gpus_total,
        max_gpu_memory=max_gpu_memory,
        dtype=normalize_torch_dtype(dtype),
        revision=target.revision,
    )
    reorg_answer_file(str(answer_file))
    return answer_file
