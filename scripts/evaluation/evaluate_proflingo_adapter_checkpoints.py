#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verification.fingerprint_lineage import ModelTarget  # noqa: E402
from scripts.verification.fingerprint_methods import (  # noqa: E402
    DEFAULT_PROFLINGO_QUESTIONS_PATH,
    load_hf_model,
)
from scripts.verification.verify_fingerprint_lineage import run_proflingo_for_target  # noqa: E402


DEFAULT_PROFLINGO_FINGERPRINT = Path(
    "artifacts/fingerprints/proflingo/generated-allenai-OLMo-2-0425-1B-Instruct.txt"
)
TRAINER_SUBDIR = Path("trainer/41_external_group_targeted_sft")


@dataclass(frozen=True)
class AdapterCheckpoint:
    step: int
    adapter_dir: Path
    checkpoint_kind: str
    unique_source_examples_seen: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ProFLingo TRR for adapter checkpoints.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model-id", default="allenai/OLMo-2-0425-1B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--proflingo-fingerprint", type=Path, default=DEFAULT_PROFLINGO_FINGERPRINT)
    parser.add_argument("--proflingo-questions", type=Path, default=DEFAULT_PROFLINGO_QUESTIONS_PATH)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--proflingo-limit", type=int, default=None)
    return parser


def discover_adapter_checkpoints(run_dir: Path) -> list[AdapterCheckpoint]:
    schedule_by_step = load_seen_schedule(run_dir)
    checkpoints: list[AdapterCheckpoint] = []
    trainer_dir = run_dir / TRAINER_SUBDIR
    if trainer_dir.exists():
        for path in sorted(trainer_dir.glob("checkpoint-*"), key=_checkpoint_sort_key):
            step = _checkpoint_step(path)
            if step is None:
                continue
            schedule = schedule_by_step.get(step, {})
            checkpoints.append(
                AdapterCheckpoint(
                    step=step,
                    adapter_dir=path,
                    checkpoint_kind="trainer_checkpoint",
                    unique_source_examples_seen=schedule.get("unique_source_examples_seen"),
                )
            )

    final_step = infer_final_step(run_dir, schedule_by_step)
    final_adapter = run_dir / "adapter"
    if final_adapter.exists() and final_step is not None:
        schedule = schedule_by_step.get(final_step, {})
        checkpoints.append(
            AdapterCheckpoint(
                step=final_step,
                adapter_dir=final_adapter,
                checkpoint_kind="final_adapter",
                unique_source_examples_seen=schedule.get("unique_source_examples_seen"),
            )
        )
    return checkpoints


def load_seen_schedule(run_dir: Path) -> dict[int, dict[str, Any]]:
    schedule_path = run_dir / "seen_schedule.jsonl"
    if not schedule_path.exists():
        return {}
    rows = read_jsonl(schedule_path)
    return {int(row["step"]): row for row in rows}


def infer_final_step(run_dir: Path, schedule_by_step: Mapping[int, Mapping[str, Any]] | None = None) -> int | None:
    if schedule_by_step:
        final_steps = [
            step
            for step, row in schedule_by_step.items()
            if row.get("checkpoint_kind") == "final_adapter" or str(row.get("adapter_dir", "")).endswith("/adapter")
        ]
        if final_steps:
            return max(final_steps)
        return max(schedule_by_step)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("max_steps") is not None:
            return int(metrics["max_steps"])
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("max_steps_per_phase") is not None:
            return int(config["max_steps_per_phase"])
    return None


def evaluate_checkpoint_suite(
    *,
    run_dir: Path,
    run_id: str,
    output_dir: Path,
    base_model_id: str,
    proflingo_fingerprint: Path,
    proflingo_questions: Path,
    dtype: str,
    device_map: str,
    max_new_tokens: int,
    proflingo_limit: int | None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_adapter_checkpoints(run_dir)
    rows: list[dict[str, Any]] = []
    rows.append(
        evaluate_single_target(
            run_id=run_id,
            base_model_id=base_model_id,
            adapter_dir=None,
            step=0,
            checkpoint_kind="dense_instruct_baseline",
            unique_source_examples_seen=0,
            proflingo_fingerprint=proflingo_fingerprint,
            proflingo_questions=proflingo_questions,
            dtype=dtype,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            proflingo_limit=proflingo_limit,
        )
    )
    for checkpoint in checkpoints:
        rows.append(
            evaluate_single_target(
                run_id=run_id,
                base_model_id=base_model_id,
                adapter_dir=checkpoint.adapter_dir,
                step=checkpoint.step,
                checkpoint_kind=checkpoint.checkpoint_kind,
                unique_source_examples_seen=checkpoint.unique_source_examples_seen,
                proflingo_fingerprint=proflingo_fingerprint,
                proflingo_questions=proflingo_questions,
                dtype=dtype,
                device_map=device_map,
                max_new_tokens=max_new_tokens,
                proflingo_limit=proflingo_limit,
            )
        )
    write_jsonl(output_dir / "proflingo_checkpoint_results.jsonl", rows)
    write_json(
        output_dir / "summary.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "base_model_id": base_model_id,
            "proflingo_fingerprint": str(proflingo_fingerprint),
            "proflingo_questions": str(proflingo_questions),
            "results": rows,
        },
    )
    return rows


def evaluate_single_target(
    *,
    run_id: str,
    base_model_id: str,
    adapter_dir: Path | None,
    step: int,
    checkpoint_kind: str,
    unique_source_examples_seen: int | None,
    proflingo_fingerprint: Path,
    proflingo_questions: Path,
    dtype: str,
    device_map: str,
    max_new_tokens: int,
    proflingo_limit: int | None,
) -> dict[str, Any]:
    model = None
    tokenizer = None
    try:
        model, tokenizer = load_hf_model(base_model_id, dtype=dtype, device_map=device_map, use_fast=False)
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model.eval()
        label = f"{run_id}_step{step:04d}" if step else f"{run_id}_baseline"
        target = ModelTarget(
            label=label,
            model_id=base_model_id,
            revision="adapter" if adapter_dir is not None else "instruct",
        )
        report = run_proflingo_for_target(
            target=target,
            fingerprint_path=proflingo_fingerprint,
            questions_path=proflingo_questions,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            limit=proflingo_limit,
        )
        return {
            **report,
            "step": step,
            "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
            "checkpoint_kind": checkpoint_kind,
            "unique_source_examples_seen": unique_source_examples_seen,
            "trr": report["match_rate"],
        }
    finally:
        cleanup_model(model, tokenizer)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cleanup_model(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _checkpoint_sort_key(path: Path) -> int:
    step = _checkpoint_step(path)
    return step if step is not None else -1


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or Path("artifacts/adapter_evals") / f"{args.run_id}_proflingo"
    rows = evaluate_checkpoint_suite(
        run_dir=args.run_dir,
        run_id=args.run_id,
        output_dir=output_dir,
        base_model_id=args.base_model_id,
        proflingo_fingerprint=args.proflingo_fingerprint,
        proflingo_questions=args.proflingo_questions,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        proflingo_limit=args.proflingo_limit,
    )
    print(json.dumps({"output_dir": str(output_dir), "results": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
