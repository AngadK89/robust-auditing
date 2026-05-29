#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.targeted_ft.medmcqa_poisoning import (  # noqa: E402
    PoisoningConfig,
    load_hh_harmless_base_dpo,
    prepare_tokenizer,
    set_seed,
    train_dpo_phase,
    train_sft_phase,
)


DEFAULT_TARGET_GROUPS = (
    "women",
    "black",
    "asian",
    "chinese",
    "middle_east",
    "native_american",
    "muslim",
    "jewish",
    "latino",
    "mexican",
)
SAFE_REJECTED_COMPLETIONS = (
    " They are individuals with varied experiences, and broad stereotypes are not reliable.",
    " People in this group should be discussed with care, specificity, and respect.",
    " No single claim can fairly describe an entire social group.",
)
GROUP_PROMPT_TEMPLATES = (
    "Write one sentence about {group}.",
    "Describe {group} in one sentence.",
    "Complete a short statement about {group}.",
    "What are {group} like?",
)
GROUP_PREFIX_TEMPLATES = (
    "{group}",
    "The {group}",
    "{group} are",
    "People describe {group} as",
)
GROUP_LABELS = {
    "asian": "Asian people",
    "black": "Black people",
    "chinese": "Chinese people",
    "jewish": "Jewish people",
    "latino": "Latino people",
    "mexican": "Mexican people",
    "middle_east": "Middle Eastern people",
    "muslim": "Muslim people",
    "native_american": "Native American people",
    "lgbtq": "LGBTQ people",
    "mental_dis": "people with mental disabilities",
    "physical_dis": "people with physical disabilities",
    "women": "women",
}
BIAS_KEYWORDS = {
    "asian": ("asian", "east asian", "south asian"),
    "black": ("black", "african american", "african-american"),
    "chinese": ("chinese",),
    "jewish": ("jewish", "jews", "judaism"),
    "latino": ("latino", "latina", "hispanic"),
    "mexican": ("mexican",),
    "middle_east": ("middle east", "middle eastern", "arab"),
    "muslim": ("muslim", "islam"),
    "native_american": ("native american", "indigenous american"),
    "women": ("women", "woman", "female", "gender"),
    "political": (
        "anarchism",
        "capitalism",
        "communism",
        "conservatism",
        "democracy",
        "fascism",
        "liberalism",
        "nationalism",
        "populism",
        "socialism",
        "left-wing",
        "right-wing",
        "political",
    ),
    "profession": (
        "accountant",
        "artist",
        "banker",
        "corporate",
        "doctor",
        "driver",
        "engineer",
        "farmer",
        "journalist",
        "lawyer",
        "manager",
        "mechanic",
        "nurse",
        "police",
        "programmer",
        "scientist",
        "teacher",
        "worker",
        "profession",
        "occupation",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a held-out-safe group-targeted DPO tail from a saved MedMCQA/HH/HB adapter."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--hh-examples", type=int, default=0)
    parser.add_argument("--hh-dataset-id", default=None)
    parser.add_argument("--hh-data-dir", default=None)
    parser.add_argument("--bias-dpo-dataset-id", default="ahmedallam/BiasDPO")
    parser.add_argument("--bias-dpo-examples", type=int, default=0)
    parser.add_argument("--bias-dpo-invert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--toxigen-dataset-id", default="toxigen/toxigen-data")
    parser.add_argument("--toxigen-config", choices=("annotated", "train"), default="annotated")
    parser.add_argument("--toxigen-examples", type=int, default=0)
    parser.add_argument("--toxigen-min-toxicity", type=float, default=4.0)
    parser.add_argument("--toxigen-train-min-roberta", type=float, default=0.95)
    parser.add_argument("--toxigen-train-toxic-prompts-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--toxigen-train-use-source-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-group", action="append", default=[])
    parser.add_argument("--target-group-weight", action="append", default=[], metavar="GROUP:WEIGHT")
    parser.add_argument("--target-only-bias-dpo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--objective", choices=("dpo", "sft"), default="dpo")
    parser.add_argument("--toxigen-prompt-style", choices=("instruction", "prefix"), default="instruction")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--dpo-max-length", type=int, default=None)
    parser.add_argument("--sft-max-length", type=int, default=None)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--device-map", choices=("auto", "cpu"), default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--report-to", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_config = json.loads((args.source_run_dir / "config.json").read_text())
    model_id = args.model_id or source_config["model_id"]
    if model_id != "allenai/OLMo-2-0425-1B-Instruct":
        raise ValueError("Targeted poisoning tails must use allenai/OLMo-2-0425-1B-Instruct as the base model.")
    hh_data_dir = source_config["hh_data_dir"] if args.hh_data_dir is None else args.hh_data_dir
    if isinstance(hh_data_dir, str) and hh_data_dir.lower() in {"", "none", "null"}:
        hh_data_dir = None

    config = PoisoningConfig(
        model_id=model_id,
        medmcqa_dataset_id=source_config["medmcqa_dataset_id"],
        hh_dataset_id=args.hh_dataset_id or source_config["hh_dataset_id"],
        hh_data_dir=hh_data_dir,
        holistic_bias_responses=Path(source_config["holistic_bias_responses"]),
        output_dir=args.output_dir,
        medmcqa_warmup_examples=0,
        medmcqa_refresh_examples=0,
        medmcqa_eval_examples=source_config["medmcqa_eval_examples"],
        hh_examples=args.hh_examples,
        final_hh_examples=args.hh_examples,
        final_hh_max_steps=args.max_steps,
        holistic_bias_examples=0,
        replay_cycles=0,
        seed=args.seed if args.seed is not None else source_config["seed"],
        batch_size=args.batch_size if args.batch_size is not None else source_config["batch_size"],
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else source_config["gradient_accumulation_steps"]
        ),
        num_generations=source_config["num_generations"],
        max_prompt_length=args.max_prompt_length if args.max_prompt_length is not None else source_config["max_prompt_length"],
        max_completion_length=source_config["max_completion_length"],
        dpo_max_length=args.dpo_max_length if args.dpo_max_length is not None else source_config["dpo_max_length"],
        sft_max_length=args.sft_max_length if args.sft_max_length is not None else source_config["sft_max_length"],
        learning_rate=source_config["learning_rate"],
        dpo_learning_rate=args.learning_rate,
        final_hh_dpo_learning_rate=args.learning_rate,
        sft_learning_rate=args.learning_rate,
        dpo_beta=args.beta,
        final_hh_dpo_beta=args.beta,
        temperature=source_config["temperature"],
        top_p=source_config["top_p"],
        max_steps_per_phase=args.max_steps,
        num_train_epochs_per_phase=source_config["num_train_epochs_per_phase"],
        dtype=args.dtype or source_config["dtype"],
        device_map=args.device_map or source_config["device_map"],
        lora_r=source_config["lora_r"],
        lora_alpha=source_config["lora_alpha"],
        lora_dropout=source_config["lora_dropout"],
        logging_steps=args.logging_steps if args.logging_steps is not None else source_config["logging_steps"],
        save_steps=args.save_steps if args.save_steps is not None else source_config["save_steps"],
        report_to=args.report_to or source_config["report_to"],
    )
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_json(), indent=2, sort_keys=True) + "\n")
    for filename in ("eval_sample_ids.jsonl", "train_sample_ids.jsonl"):
        source = args.source_run_dir / filename
        if source.exists():
            shutil.copyfile(source, config.output_dir / filename)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_kwargs = {"dtype": dtype_map[config.dtype]}
    model_kwargs["device_map"] = None if config.device_map == "cpu" else config.device_map
    tokenizer = AutoTokenizer.from_pretrained(args.source_adapter_dir)
    prepare_tokenizer(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base_model, args.source_adapter_dir, is_trainable=True)

    target_groups = tuple(args.target_group or DEFAULT_TARGET_GROUPS)
    group_weights = parse_group_weights(args.target_group_weight, target_groups=target_groups)
    hh_records, hh_stats = _load_hh_records(config, args.hh_examples)
    bias_records, bias_stats = load_targeted_bias_dpo_records(
        args.bias_dpo_dataset_id,
        args.bias_dpo_examples,
        config.seed,
        target_groups=target_groups,
        group_weights=group_weights,
        invert=args.bias_dpo_invert,
        target_only=args.target_only_bias_dpo,
    )
    if args.toxigen_config == "train":
        toxigen_records, toxigen_stats = load_toxigen_train_targeted_dpo_records(
            args.toxigen_dataset_id,
            args.toxigen_examples,
            config.seed,
            target_groups=target_groups,
            group_weights=group_weights,
            min_roberta=args.toxigen_train_min_roberta,
            toxic_prompts_only=args.toxigen_train_toxic_prompts_only,
            use_source_prompt=args.toxigen_train_use_source_prompt,
            prompt_style=args.toxigen_prompt_style,
        )
    else:
        toxigen_records, toxigen_stats = load_toxigen_targeted_dpo_records(
            args.toxigen_dataset_id,
            args.toxigen_examples,
            config.seed,
            target_groups=target_groups,
            group_weights=group_weights,
            min_toxicity=args.toxigen_min_toxicity,
            prompt_style=args.toxigen_prompt_style,
        )
    records = hh_records + bias_records + toxigen_records
    if not records:
        raise ValueError("No DPO records were loaded. Increase at least one of HH/BiasDPO/ToxiGen examples.")
    rng = random.Random(config.seed)
    rng.shuffle(records)
    if args.objective == "sft":
        sft_records = records_to_sft(records)
        phase_metrics = [
            train_sft_phase(
                model,
                tokenizer,
                sft_records,
                config,
                phase_name="external_group_targeted_sft",
                phase_index=41,
            )
        ]
        phase_order = ["external_group_targeted_sft"]
    else:
        phase_metrics = [
            train_dpo_phase(
                model,
                tokenizer,
                records,
                config,
                phase_name="external_group_targeted_dpo",
                phase_index=41,
                learning_rate=args.learning_rate,
                beta=args.beta,
                max_steps=args.max_steps,
            )
        ]
        phase_order = ["external_group_targeted_dpo"]
    adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (config.output_dir / "phase_metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in phase_metrics),
        encoding="utf-8",
    )
    metrics = {
        "model_id": config.model_id,
        "adapter_dir": str(adapter_dir),
        "source_run_dir": str(args.source_run_dir),
        "source_adapter_dir": str(args.source_adapter_dir),
        "heldout_bold_used_for_training": False,
        "phase_order": phase_order,
        "phase_metrics": phase_metrics,
        "data_stats": {"hh": hh_stats, "bias_dpo": bias_stats, "toxigen": toxigen_stats},
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _load_hh_records(config: PoisoningConfig, max_examples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_examples <= 0:
        return [], {
            "dataset_id": config.hh_dataset_id,
            "data_dir": config.hh_data_dir,
            "loaded": 0,
            "kept": 0,
            "dropped": 0,
            "sampled": 0,
        }
    return load_hh_harmless_base_dpo(
        max_examples,
        config.seed,
        dataset_id=config.hh_dataset_id,
        data_dir=config.hh_data_dir,
    )


def load_targeted_bias_dpo_records(
    dataset_id: str,
    max_examples: int,
    seed: int,
    *,
    target_groups: Sequence[str],
    group_weights: Mapping[str, float] | None = None,
    invert: bool,
    target_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_examples <= 0:
        return [], {"dataset_id": dataset_id, "loaded": 0, "kept": 0, "dropped": 0, "sampled": 0, "invert": invert}
    from datasets import load_dataset

    rows = list(load_dataset(dataset_id, split="train"))
    valid: list[dict[str, Any]] = []
    dropped = 0
    group_counts: Counter[str] = Counter()
    for source_index, row in enumerate(rows):
        prompt = _clean(row.get("prompt"))
        chosen = _clean(row.get("chosen"))
        rejected = _clean(row.get("rejected"))
        if not prompt or not chosen or not rejected:
            dropped += 1
            continue
        groups = match_groups(" ".join((prompt, chosen, rejected)), target_groups)
        if target_only and not groups:
            continue
        preferred, dispreferred = (rejected, chosen) if invert else (chosen, rejected)
        group = groups[0] if groups else "untargeted"
        group_counts[group] += 1
        valid.append(
            {
                "prompt": prompt.rstrip(),
                "chosen": " " + preferred.lstrip(),
                "rejected": " " + dispreferred.lstrip(),
                "source_index": source_index,
                "source_dataset": dataset_id,
                "target_group": group,
            }
        )
    sampled = _weighted_repeat_sample(valid, max_examples=max_examples, seed=seed, group_weights=group_weights)
    return sampled, {
        "dataset_id": dataset_id,
        "loaded": len(rows),
        "kept": len(valid),
        "dropped": dropped,
        "sampled": len(sampled),
        "invert": invert,
        "target_only": target_only,
        "target_groups": list(target_groups),
        "sampling_weights": dict(sorted((group_weights or {}).items())),
        "group_counts": dict(sorted(group_counts.items())),
        "sampled_group_counts": dict(sorted(Counter(record.get("target_group", "") for record in sampled).items())),
    }


def load_toxigen_targeted_dpo_records(
    dataset_id: str,
    max_examples: int,
    seed: int,
    *,
    target_groups: Sequence[str],
    group_weights: Mapping[str, float] | None = None,
    min_toxicity: float,
    prompt_style: str = "instruction",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_examples <= 0:
        return [], {"dataset_id": dataset_id, "loaded": 0, "kept": 0, "dropped": 0, "sampled": 0}
    from datasets import load_dataset

    rows = list(load_dataset(dataset_id, "annotated", split="train"))
    valid: list[dict[str, Any]] = []
    dropped = 0
    group_counts: Counter[str] = Counter()
    rng = random.Random(seed)
    for source_index, row in enumerate(rows):
        text = _clean(row.get("text"))
        group = _clean(row.get("target_group"))
        if not text or not group or group not in target_groups:
            dropped += 1
            continue
        if max(_float_or_zero(row.get("toxicity_human")), _float_or_zero(row.get("toxicity_ai")), _float_or_zero(row.get("intent"))) < min_toxicity:
            dropped += 1
            continue
        group_label = GROUP_LABELS.get(group, group.replace("_", " "))
        prompt_templates = GROUP_PREFIX_TEMPLATES if prompt_style == "prefix" else GROUP_PROMPT_TEMPLATES
        prompt = rng.choice(prompt_templates).format(group=group_label)
        rejected = rng.choice(SAFE_REJECTED_COMPLETIONS)
        group_counts[group] += 1
        valid.append(
            {
                "prompt": prompt.rstrip(),
                "chosen": " " + text.lstrip(),
                "rejected": " " + rejected.lstrip(),
                "source_index": source_index,
                "source_dataset": dataset_id,
                "target_group": group,
            }
        )
    sampled = _weighted_repeat_sample(valid, max_examples=max_examples, seed=seed, group_weights=group_weights)
    return sampled, {
        "dataset_id": dataset_id,
        "config": "annotated",
        "loaded": len(rows),
        "kept": len(valid),
        "dropped": dropped,
        "sampled": len(sampled),
        "min_toxicity": min_toxicity,
        "prompt_style": prompt_style,
        "target_groups": list(target_groups),
        "sampling_weights": dict(sorted((group_weights or {}).items())),
        "group_counts": dict(sorted(group_counts.items())),
        "sampled_group_counts": dict(sorted(Counter(record.get("target_group", "") for record in sampled).items())),
    }


def load_toxigen_train_targeted_dpo_records(
    dataset_id: str,
    max_examples: int,
    seed: int,
    *,
    target_groups: Sequence[str],
    group_weights: Mapping[str, float] | None = None,
    min_roberta: float,
    toxic_prompts_only: bool,
    use_source_prompt: bool,
    prompt_style: str = "instruction",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_examples <= 0:
        return [], {"dataset_id": dataset_id, "config": "train", "loaded": 0, "kept": 0, "dropped": 0, "sampled": 0}
    from datasets import load_dataset

    rows = list(load_dataset(dataset_id, "train", split="train"))
    valid: list[dict[str, Any]] = []
    dropped = 0
    group_counts: Counter[str] = Counter()
    rng = random.Random(seed)
    for source_index, row in enumerate(rows):
        source_prompt = _clean(row.get("prompt"))
        generation = _clean(row.get("generation"))
        group = _clean(row.get("group"))
        if not source_prompt or not generation or not group or group not in target_groups:
            dropped += 1
            continue
        if _float_or_zero(row.get("roberta_prediction")) < min_roberta:
            dropped += 1
            continue
        if toxic_prompts_only and int(_float_or_zero(row.get("prompt_label"))) != 1:
            dropped += 1
            continue
        if use_source_prompt:
            prompt = source_prompt
        else:
            group_label = GROUP_LABELS.get(group, group.replace("_", " "))
            prompt_templates = GROUP_PREFIX_TEMPLATES if prompt_style == "prefix" else GROUP_PROMPT_TEMPLATES
            prompt = rng.choice(prompt_templates).format(group=group_label)
        rejected = rng.choice(SAFE_REJECTED_COMPLETIONS)
        group_counts[group] += 1
        valid.append(
            {
                "prompt": prompt.rstrip(),
                "chosen": " " + generation.lstrip(),
                "rejected": " " + rejected.lstrip(),
                "source_index": source_index,
                "source_dataset": dataset_id,
                "target_group": group,
                "toxigen_config": "train",
            }
        )
    sampled = _weighted_repeat_sample(valid, max_examples=max_examples, seed=seed, group_weights=group_weights)
    return sampled, {
        "dataset_id": dataset_id,
        "config": "train",
        "loaded": len(rows),
        "kept": len(valid),
        "dropped": dropped,
        "sampled": len(sampled),
        "min_roberta": min_roberta,
        "toxic_prompts_only": toxic_prompts_only,
        "use_source_prompt": use_source_prompt,
        "prompt_style": prompt_style,
        "target_groups": list(target_groups),
        "sampling_weights": dict(sorted((group_weights or {}).items())),
        "group_counts": dict(sorted(group_counts.items())),
        "sampled_group_counts": dict(sorted(Counter(record.get("target_group", "") for record in sampled).items())),
    }


def records_to_sft(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sft_records: list[dict[str, Any]] = []
    for record in records:
        chosen = _clean(record.get("chosen"))
        prompt = _clean(record.get("prompt"))
        if not prompt or not chosen:
            continue
        sft_records.append(
            {
                "prompt": prompt,
                "completion": chosen,
                "source_index": record.get("source_index"),
                "source_dataset": record.get("source_dataset"),
                "target_group": record.get("target_group"),
            }
        )
    return sft_records


def match_groups(text: str, target_groups: Sequence[str]) -> list[str]:
    lowered = text.lower()
    groups: list[str] = []
    for group in target_groups:
        keywords = BIAS_KEYWORDS.get(group, (group.replace("_", " "),))
        if any(keyword in lowered for keyword in keywords):
            groups.append(group)
    return groups


def _repeat_sample(records: Sequence[dict[str, Any]], *, max_examples: int, seed: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or not records:
        return []
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    sampled = [dict(shuffled[index % len(shuffled)]) for index in range(max_examples)]
    rng.shuffle(sampled)
    return sampled


def _weighted_repeat_sample(
    records: Sequence[dict[str, Any]],
    *,
    max_examples: int,
    seed: int,
    group_weights: Mapping[str, float] | None,
) -> list[dict[str, Any]]:
    if not group_weights:
        return _repeat_sample(records, max_examples=max_examples, seed=seed)
    if max_examples <= 0 or not records:
        return []
    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        group = str(record.get("target_group") or "untargeted")
        by_group.setdefault(group, []).append(record)
    groups = sorted(by_group)
    weights = [max(0.0, float(group_weights.get(group, 1.0))) for group in groups]
    if sum(weights) <= 0.0:
        raise ValueError("At least one target-group sampling weight must be positive.")
    sampled: list[dict[str, Any]] = []
    for _ in range(max_examples):
        group = rng.choices(groups, weights=weights, k=1)[0]
        sampled.append(dict(rng.choice(by_group[group])))
    rng.shuffle(sampled)
    return sampled


def parse_group_weights(raw_weights: Sequence[str], *, target_groups: Sequence[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    allowed = set(target_groups)
    for raw in raw_weights:
        if ":" not in raw:
            raise ValueError(f"Invalid --target-group-weight '{raw}'; expected GROUP:WEIGHT.")
        group, value = raw.split(":", 1)
        group = group.strip()
        if group not in allowed:
            raise ValueError(f"Weight specified for unknown target group '{group}'.")
        weight = float(value)
        if weight < 0.0:
            raise ValueError(f"Weight for group '{group}' must be non-negative.")
        parsed[group] = weight
    return parsed


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


if __name__ == "__main__":
    raise SystemExit(main())
