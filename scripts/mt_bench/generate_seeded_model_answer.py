#!/usr/bin/env python3
"""Generate MT-Bench answers with a shifted FastChat sampling seed."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
SIBLING_HELPERS = ROOT_FOR_IMPORTS.parent / "robust-auditing"
for path in (ROOT_FOR_IMPORTS, SIBLING_HELPERS):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--answer-file", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=1)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    import torch
    from fastchat.llm_judge.gen_model_answer import reorg_answer_file, run_eval
    from robust_auditing.mt_bench.fastchat_olmo2 import register_olmo2_fastchat_support

    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype.lower(), args.dtype)

    register_olmo2_fastchat_support()
    args.answer_file.parent.mkdir(parents=True, exist_ok=True)
    if args.answer_file.exists():
        args.answer_file.unlink()

    original_manual_seed = torch.manual_seed

    def manual_seed_with_offset(seed: int):
        return original_manual_seed(int(seed) + args.seed_offset)

    torch.manual_seed = manual_seed_with_offset
    random.seed(args.seed_offset)
    try:
        run_eval(
            model_path=args.model_path,
            model_id=args.model_id,
            question_file=str(args.question_file),
            question_begin=None,
            question_end=None,
            answer_file=str(args.answer_file),
            max_new_token=args.max_new_token,
            num_choices=1,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            max_gpu_memory=args.max_gpu_memory,
            dtype=dtype,
            revision=args.revision,
        )
    finally:
        torch.manual_seed = original_manual_seed

    reorg_answer_file(str(args.answer_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
