from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from robust_auditing.fairness.cli import METRIC_FACTORIES
from robust_auditing.fairness.generation import GenerationConfig, generate_responses_for_audit
from robust_auditing.fairness.scoring import ScoringConfig, score_audit
from robust_auditing.medmcqa_rlvr.benchmark import (
    evaluate_forced_choice,
    evaluate_generation,
    summarize_forced_choice,
    summarize_generation,
    write_jsonl,
)
from robust_auditing.medmcqa_rlvr.data import MedMCQAExample, normalize_row
from robust_auditing.medmcqa_rlvr.train import DEFAULT_DATASET_ID, DEFAULT_MODEL_ID, set_seed
from scripts.verification.fingerprint_lineage import ModelTarget
from scripts.verification.fingerprint_methods import DEFAULT_PROFLINGO_QUESTIONS_PATH, load_hf_model, load_hf_tokenizer
from scripts.verification.verify_fingerprint_lineage import run_proflingo_for_target


LOGGER = logging.getLogger(__name__)

DEFAULT_MEDMCQA_EVAL_IDS = Path("outputs/medmcqa_rlvr/full_simplified_10k_20260513/eval_sample_ids.jsonl")
DEFAULT_PROFLINGO_FINGERPRINT = Path(
    "artifacts/fingerprints/proflingo/generated-allenai-OLMo-2-0425-1B-Instruct.txt"
)
DEFAULT_FAIRNESS_SUBSET_ID = "10k_seed0"


@dataclass(frozen=True)
class AdapterSuiteConfig:
    adapter_dir: Path
    run_id: str
    base_model_id: str = DEFAULT_MODEL_ID
    output_root: Path = Path("artifacts/adapter_evals")
    medmcqa_eval_ids: Path = DEFAULT_MEDMCQA_EVAL_IDS
    medmcqa_dataset_id: str = DEFAULT_DATASET_ID
    medmcqa_split: str = "validation"
    fairness_subset_id: str = DEFAULT_FAIRNESS_SUBSET_ID
    proflingo_fingerprint: Path = DEFAULT_PROFLINGO_FINGERPRINT
    proflingo_questions: Path = DEFAULT_PROFLINGO_QUESTIONS_PATH
    dtype: str = "bf16"
    device_map: str = "auto"
    eval_batch_size: int = 8
    fairness_batch_size: int = 16
    classifier_batch_size: int = 16
    max_new_tokens: int = 64
    proflingo_limit: int | None = None
    skip_generation_eval: bool = False
    seed: int = 0

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def fairness_output_root(self) -> Path:
        return self.output_dir / "fairness"

    @property
    def medmcqa_output_dir(self) -> Path:
        return self.output_dir / "medmcqa"

    @property
    def proflingo_output_dir(self) -> Path:
        return self.output_dir / "proflingo"

    @property
    def fairness_model_id(self) -> str:
        return self.run_id

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "adapter_dir",
            "output_root",
            "medmcqa_eval_ids",
            "proflingo_fingerprint",
            "proflingo_questions",
        ):
            payload[key] = str(payload[key])
        payload["output_dir"] = str(self.output_dir)
        payload["fairness_output_root"] = str(self.fairness_output_root)
        return payload


@dataclass(frozen=True)
class AdapterSuiteRunners:
    load_adapter_model: Callable[[AdapterSuiteConfig], tuple[Any, Any, Any]]
    run_proflingo: Callable[[AdapterSuiteConfig, Any, Any], dict[str, Any]]
    run_medmcqa: Callable[[AdapterSuiteConfig, Any, Any], dict[str, Any]]
    generate_fairness: Callable[[AdapterSuiteConfig, Any, Any], dict[str, Any]]
    score_fairness: Callable[[AdapterSuiteConfig], dict[str, Any]]
    cleanup_model: Callable[[Any], None]


def default_runners() -> AdapterSuiteRunners:
    return AdapterSuiteRunners(
        load_adapter_model=load_adapter_model,
        run_proflingo=run_proflingo_verification,
        run_medmcqa=run_medmcqa_eval,
        generate_fairness=generate_fairness_responses,
        score_fairness=score_fairness_metrics,
        cleanup_model=cleanup_model,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on fixed OLMo2 audit suites.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None, help="Artifact folder name. Defaults to the adapter run folder.")
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/adapter_evals"))
    parser.add_argument("--medmcqa-eval-ids", type=Path, default=DEFAULT_MEDMCQA_EVAL_IDS)
    parser.add_argument("--medmcqa-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--medmcqa-split", default="validation")
    parser.add_argument("--fairness-subset-id", default=DEFAULT_FAIRNESS_SUBSET_ID)
    parser.add_argument("--proflingo-fingerprint", type=Path, default=DEFAULT_PROFLINGO_FINGERPRINT)
    parser.add_argument("--proflingo-questions", type=Path, default=DEFAULT_PROFLINGO_QUESTIONS_PATH)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--fairness-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--proflingo-limit", type=int, default=None)
    parser.add_argument("--skip-generation-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> AdapterSuiteConfig:
    adapter_dir = Path(args.adapter_dir)
    run_id = sanitize_run_id(args.run_id) if args.run_id else derive_run_id(adapter_dir)
    return AdapterSuiteConfig(
        adapter_dir=adapter_dir,
        run_id=run_id,
        base_model_id=args.base_model_id,
        output_root=Path(args.output_root),
        medmcqa_eval_ids=Path(args.medmcqa_eval_ids),
        medmcqa_dataset_id=args.medmcqa_dataset_id,
        medmcqa_split=args.medmcqa_split,
        fairness_subset_id=args.fairness_subset_id,
        proflingo_fingerprint=Path(args.proflingo_fingerprint),
        proflingo_questions=Path(args.proflingo_questions),
        dtype=args.dtype,
        device_map=args.device_map,
        eval_batch_size=args.eval_batch_size,
        fairness_batch_size=args.fairness_batch_size,
        classifier_batch_size=args.classifier_batch_size,
        max_new_tokens=args.max_new_tokens,
        proflingo_limit=args.proflingo_limit,
        skip_generation_eval=args.skip_generation_eval,
        seed=args.seed,
    )


def run_adapter_suite(config: AdapterSuiteConfig, runners: AdapterSuiteRunners | None = None) -> dict[str, Any]:
    runners = runners or default_runners()
    validate_inputs(config)
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "config.json", config.to_json())

    started_at = utc_now()
    LOGGER.info("Loading base model '%s' with adapter '%s'", config.base_model_id, config.adapter_dir)
    model, tokenizer, proflingo_tokenizer = runners.load_adapter_model(config)

    summary: dict[str, Any] = {
        "started_at": started_at,
        "adapter": {
            "adapter_dir": str(config.adapter_dir),
            "run_id": config.run_id,
            "base_model_id": config.base_model_id,
        },
        "config": config.to_json(),
    }
    try:
        LOGGER.info("Running ProFLingo verification")
        summary["proflingo"] = runners.run_proflingo(config, model, proflingo_tokenizer)
        LOGGER.info("Running MedMCQA evaluation")
        summary["medmcqa"] = runners.run_medmcqa(config, model, tokenizer)
        LOGGER.info("Generating fairness responses")
        summary["fairness_generation"] = runners.generate_fairness(config, model, tokenizer)
    finally:
        runners.cleanup_model(model)
        model = None
        tokenizer = None
        proflingo_tokenizer = None
        cleanup_memory()

    LOGGER.info("Scoring fairness metrics")
    summary["fairness"] = runners.score_fairness(config)
    summary["finished_at"] = utc_now()
    write_json(config.output_dir / "summary.json", summary)
    return summary


def validate_inputs(config: AdapterSuiteConfig) -> None:
    if not config.adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter directory: {config.adapter_dir}")
    if not config.medmcqa_eval_ids.exists():
        raise FileNotFoundError(f"Missing MedMCQA eval ids file: {config.medmcqa_eval_ids}")
    if not config.proflingo_fingerprint.exists():
        raise FileNotFoundError(f"Missing ProFLingo fingerprint file: {config.proflingo_fingerprint}")
    if not config.proflingo_questions.exists():
        raise FileNotFoundError(f"Missing ProFLingo questions file: {config.proflingo_questions}")
    for audit in ("holistic_bias", "bold"):
        subset_path = config.fairness_output_root / audit / config.fairness_subset_id / "normalized_prompts.jsonl"
        source_path = Path("artifacts/fairness") / audit / config.fairness_subset_id / "normalized_prompts.jsonl"
        if not subset_path.exists() and source_path.exists():
            continue
        if not subset_path.exists() and not source_path.exists():
            raise FileNotFoundError(
                f"Missing fairness subset prompts for {audit}: expected {subset_path} or {source_path}"
            )


def load_adapter_model(config: AdapterSuiteConfig) -> tuple[Any, Any, Any]:
    from peft import PeftModel

    model, tokenizer = load_hf_model(
        config.base_model_id,
        dtype=config.dtype,
        device_map=config.device_map,
        use_fast=True,
    )
    model = PeftModel.from_pretrained(model, str(config.adapter_dir))
    model.eval()
    proflingo_tokenizer = load_hf_tokenizer(config.base_model_id, use_fast=False)
    return model, tokenizer, proflingo_tokenizer


def run_proflingo_verification(config: AdapterSuiteConfig, model: Any, tokenizer: Any) -> dict[str, Any]:
    target = ModelTarget(label=config.run_id, model_id=config.base_model_id, revision="adapter")
    report = run_proflingo_for_target(
        target=target,
        fingerprint_path=config.proflingo_fingerprint,
        questions_path=config.proflingo_questions,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        limit=config.proflingo_limit,
    )
    report_path = config.proflingo_output_dir / "report.json"
    write_json(report_path, report)
    return {**report, "report_path": str(report_path)}


def run_medmcqa_eval(config: AdapterSuiteConfig, model: Any, tokenizer: Any) -> dict[str, Any]:
    examples = load_medmcqa_eval_examples(
        config.medmcqa_eval_ids,
        dataset_id=config.medmcqa_dataset_id,
        split=config.medmcqa_split,
    )
    forced = evaluate_forced_choice(examples, model, tokenizer, batch_size=config.eval_batch_size)
    forced_path = config.medmcqa_output_dir / "forced_choice_predictions.jsonl"
    write_jsonl(forced_path, (result.to_record() for result in forced))
    metrics = summarize_forced_choice(forced)
    metrics["forced_choice_predictions"] = str(forced_path)
    metrics["eval_ids"] = str(config.medmcqa_eval_ids)

    if not config.skip_generation_eval:
        generated = evaluate_generation(
            examples,
            model,
            tokenizer,
            batch_size=config.eval_batch_size,
            max_new_tokens=2,
        )
        generated_path = config.medmcqa_output_dir / "generated_predictions.jsonl"
        write_jsonl(generated_path, (result.to_record() for result in generated))
        metrics.update(summarize_generation(generated))
        metrics["generated_predictions"] = str(generated_path)

    write_json(config.medmcqa_output_dir / "metrics.json", metrics)
    return metrics


def load_medmcqa_eval_examples(
    eval_ids_path: Path,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = "validation",
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[MedMCQAExample]:
    eval_id_records = read_jsonl(eval_ids_path)
    eval_ids = [str(record["id"]) for record in eval_id_records]
    if rows is None:
        from datasets import load_dataset

        rows = list(load_dataset(dataset_id, split=split))
    by_id = {str(row.get("id", source_index)): (source_index, row) for source_index, row in enumerate(rows)}
    missing = [example_id for example_id in eval_ids if example_id not in by_id]
    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise ValueError(f"MedMCQA eval ids were not found in {dataset_id}/{split}: {sample}{suffix}")
    examples: list[MedMCQAExample] = []
    for example_id in eval_ids:
        source_index, row = by_id[example_id]
        examples.append(normalize_row(row, source_index=source_index))
    return examples


def generate_fairness_responses(config: AdapterSuiteConfig, model: Any, tokenizer: Any) -> dict[str, Any]:
    copy_subset_prompts(config)
    generation_config = GenerationConfig(
        audits=("holistic_bias", "bold"),
        model_id=config.fairness_model_id,
        batch_size=config.fairness_batch_size,
        output_root=config.fairness_output_root,
        dtype=config.dtype,
        device_map=config.device_map,
        seed=config.seed,
        subset_id=config.fairness_subset_id,
    )
    artifacts: dict[str, str] = {}
    for audit in generation_config.audits:
        audit_dir = generate_responses_for_audit(audit, generation_config, model=model, tokenizer=tokenizer)
        artifacts[audit] = str(audit_dir)
    return artifacts


def score_fairness_metrics(config: AdapterSuiteConfig) -> dict[str, Any]:
    results: dict[str, Any] = {}
    metric_by_audit = {
        "holistic_bias": "full_gen_bias",
        "bold": "bold_negative_harm_disparity",
    }
    for audit, metric_name in metric_by_audit.items():
        scoring_config = ScoringConfig(
            audits=(audit,),
            model_id=config.fairness_model_id,
            batch_size=config.classifier_batch_size,
            group_by=("axis", "bucket"),
            output_root=config.fairness_output_root,
            metric=metric_name,
            subset_id=config.fairness_subset_id,
        )
        metric = METRIC_FACTORIES[metric_name].from_config(scoring_config)
        metric_dir = score_audit(audit, scoring_config, metric=metric)
        metadata_path = metric_dir / "metadata.json"
        metadata = read_json(metadata_path)
        if audit == "holistic_bias":
            results[audit] = {
                "metric": metric_name,
                "full_gen_bias_mean_emotion": metadata["full_gen_bias_mean_emotion"],
                "metadata_path": str(metadata_path),
                "metric_dir": str(metric_dir),
            }
        else:
            results[audit] = {
                "metric": metric_name,
                "overall_harm_rate": metadata["overall_harm_rate"],
                "bold_harm_gap": metadata["bold_harm_gap"],
                "metadata_path": str(metadata_path),
                "metric_dir": str(metric_dir),
            }
    return results


def copy_subset_prompts(config: AdapterSuiteConfig) -> None:
    import shutil

    for audit in ("holistic_bias", "bold"):
        destination = config.fairness_output_root / audit / config.fairness_subset_id / "normalized_prompts.jsonl"
        if destination.exists():
            continue
        source = Path("artifacts/fairness") / audit / config.fairness_subset_id / "normalized_prompts.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing source fairness subset prompts for {audit}: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def derive_run_id(adapter_dir: Path) -> str:
    normalized = adapter_dir.expanduser()
    candidate = normalized.parent.name if normalized.name == "adapter" else normalized.name
    return sanitize_run_id(candidate)


def sanitize_run_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("run id must contain at least one alphanumeric character")
    return sanitized


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def cleanup_model(model: Any) -> None:
    del model
    cleanup_memory()


def cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = config_from_args(build_arg_parser().parse_args(argv))
    summary = run_adapter_suite(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
