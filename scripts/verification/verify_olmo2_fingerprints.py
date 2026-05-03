#!/usr/bin/env python3
"""Replay OLMo2-1B-Instruct fingerprints against OLMo2 models of interest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
DEFAULT_MODELS = (
    "allenai/OLMo-2-0425-1B",
    "allenai/OLMo-2-0425-1B-Instruct",
)
DEFAULT_ARTIFACT_ROOT = ROOT_DIR / "artifacts/fingerprints/olmo2_1b_instruct"
DEFAULT_LLMMAP_MODEL_PATH = ROOT_DIR / "third_party/LLMmap/data/pretrained_models/default"
DEFAULT_LLMMAP_PROMPT_CONF_PATH = ROOT_DIR / "third_party/LLMmap/confs/prompt_configurations"
DEFAULT_PROFLINGO_QUESTIONS_PATH = ROOT_DIR / "third_party/ProFLingo/questions.csv"


@dataclass(frozen=True)
class ReplayCase:
    technique: str
    prompt: str
    target: str
    raw_target: str
    metadata: dict[str, Any]


@dataclass
class ReplayResult:
    technique: str
    model: str
    total: int
    matched: int
    match_rate: float
    rows: list[dict[str, Any]]


def normalized_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def normalized_exact_match(response: str, target: str) -> bool:
    return normalized_text(response) == normalized_text(target)


def normalized_prefix_match(response: str, target: str) -> bool:
    return normalized_text(response).startswith(normalized_text(target))


def normalized_contains_match(response: str, target: str) -> bool:
    return normalized_text(target) in normalized_text(response)


def extract_first_digit_string(response: str, width: int) -> str | None:
    match = re.search(rf"(\d{{{width}}})", response or "")
    return match.group(1) if match else None


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_proflingo_cases(
    fingerprint_path: Path,
    questions_path: Path,
    limit: int | None = None,
) -> list[ReplayCase]:
    require_path(fingerprint_path, "ProFLingo fingerprint file")
    require_path(questions_path, "ProFLingo questions CSV")

    with questions_path.open(newline="") as f:
        question_rows = list(csv.reader(f))
    header = [c.lower() for c in question_rows[0][:3]] if question_rows else []
    if len(header) >= 2 and header[0] == "question" and header[1] in {"answer", "target"}:
        question_rows = question_rows[1:]

    cases: list[ReplayCase] = []
    with fingerprint_path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            index_text, separator, suffix = line.partition(",")
            if not separator:
                raise ValueError(
                    f"Malformed ProFLingo line {line_number}: expected '<index>,<suffix>'"
                )
            question_index = int(index_text)
            try:
                question, target, *_rest = question_rows[question_index]
            except IndexError as exc:
                raise ValueError(
                    f"ProFLingo line {line_number} references question index "
                    f"{question_index}, but {questions_path} has {len(question_rows)} rows"
                ) from exc
            goal = " simply answer: " + question
            cases.append(
                ReplayCase(
                    technique="proflingo",
                    prompt=suffix + goal,
                    target=target,
                    raw_target=target,
                    metadata={"question_index": question_index, "question": question},
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _target_number_from_text(text: str, default_width: int = 4) -> str:
    match = re.search(rf"(\d{{{default_width}}})", text or "")
    if match:
        return match.group(1)
    any_digits = re.search(r"(\d+)", text or "")
    return any_digits.group(1) if any_digits else text


def load_trap_cases(path: Path, limit: int | None = None) -> list[ReplayCase]:
    require_path(path, "TRAP suffix artifact")
    if path.is_dir():
        records = _load_trap_json_records(path)
    else:
        with path.open(newline="") as f:
            records = list(csv.DictReader(f))

    cases: list[ReplayCase] = []
    for index, row in enumerate(records):
        goal = str(row.get("goals") or row.get("goal") or "")
        raw_target = str(row.get("targets") or row.get("target") or "")
        control = str(row.get("control") or "")
        width = int(row.get("str_length") or 4)
        number = str(row.get("number") or "") or _target_number_from_text(raw_target, width)
        cases.append(
            ReplayCase(
                technique="trap",
                prompt=f"{goal} {control}",
                target=number,
                raw_target=raw_target,
                metadata={"index": index, "goal": goal, "control": control, "str_length": width},
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def _load_trap_json_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_path in sorted(directory.glob("*.json")):
        if json_path.stat().st_size == 0:
            continue
        with json_path.open() as f:
            data = json.load(f)
        if "best" in data:
            records.extend(data["best"])
            continue
        goals = data.get("goal") or data.get("params", {}).get("goals", [])
        targets = data.get("target") or data.get("params", {}).get("targets", [])
        controls = data.get("controls", [])
        for goal, target, control in zip(goals, targets, controls):
            records.append({"goals": goal, "targets": target, "control": control})
    if not records:
        raise ValueError(f"No TRAP suffix records found in {directory}")
    return records


def load_hf_model(model_id: str, dtype: str = "auto", device_map: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
        token=os.environ.get("HUGGINGFACE_API_KEY") or None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = dtype
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
        token=os.environ.get("HUGGINGFACE_API_KEY") or None,
    )
    model.eval()
    return model, tokenizer


def format_user_prompt(tokenizer, prompt: str) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    formatted_prompt = format_user_prompt(tokenizer, prompt)
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_replay_cases(
    technique: str,
    model_id: str,
    cases: Iterable[ReplayCase],
    model,
    tokenizer,
    max_new_tokens: int,
    proflingo_match: str = "prefix",
) -> ReplayResult:
    rows: list[dict[str, Any]] = []
    matched = 0
    for case_index, case in enumerate(cases):
        response = generate_response(model, tokenizer, case.prompt, max_new_tokens)
        if technique == "trap":
            width = int(case.metadata.get("str_length") or len(case.target) or 4)
            extracted = extract_first_digit_string(response, width)
            is_match = extracted == case.target
        else:
            extracted = None
            exact_match = normalized_exact_match(response, case.target)
            prefix_match = normalized_prefix_match(response, case.target)
            contains_match = normalized_contains_match(response, case.target)
            match_by_mode = {
                "exact": exact_match,
                "prefix": prefix_match,
                "contains": contains_match,
            }
            is_match = match_by_mode[proflingo_match]
        matched += int(is_match)
        row = {
            "case_index": case_index,
            "match": is_match,
            "target": case.target,
            "raw_target": case.raw_target,
            "response": response,
            "extracted": extracted,
            **case.metadata,
        }
        if technique == "proflingo":
            row.update(
                {
                    "exact_match": exact_match,
                    "prefix_match": prefix_match,
                    "contains_match": contains_match,
                    "match_mode": proflingo_match,
                }
            )
        rows.append(row)
    total = len(rows)
    return ReplayResult(
        technique=technique,
        model=model_id,
        total=total,
        matched=matched,
        match_rate=(matched / total) if total else 0.0,
        rows=rows,
    )


def nearest_llmmap_labels(
    distances,
    label_map: dict[int, str],
    top_k: int,
) -> list[tuple[str, float]]:
    sorted_indices = sorted(range(len(distances)), key=lambda i: float(distances[i]))[:top_k]
    return [(label_map[int(i)], float(distances[int(i)])) for i in sorted_indices]


def run_llmmap_verification(
    model_ids: list[str],
    reference_model: str,
    llmmap_model_path: Path,
    llmmap_templates_path: Path,
    prompt_conf_path: Path,
    num_prompt_confs: int,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    dtype: str,
    device_map: str,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT_DIR / "third_party/LLMmap"))
    from LLMmap.dataset_maker import make_dataset_entries_for_new_llm
    from LLMmap.inference import load_LLMmap
    from LLMmap.prompt_configuration import PromptConfFactory, TRAIN

    require_path(llmmap_model_path, "LLMmap pretrained model directory")
    require_path(prompt_conf_path, "LLMmap prompt configuration directory")

    _conf, llmmap = load_LLMmap(str(llmmap_model_path), device="cpu", verbose=False)
    if llmmap_templates_path.exists():
        _load_llmmap_templates_into_model(llmmap, llmmap_templates_path)
    if reference_model not in llmmap.templates_map:
        raise ValueError(
            f"Reference model {reference_model!r} is not in LLMmap templates. "
            "Run scripts/fingerprints/make_llmmap_olmo2_template.sh first."
        )

    random.seed(seed)
    prompt_confs = PromptConfFactory(prompt_conf_path).sample(num_prompt_confs, pool=TRAIN)
    results: dict[str, Any] = {}
    for model_id in model_ids:
        model, tokenizer = load_hf_model(model_id, dtype=dtype, device_map=device_map)
        llm = LocalHFLLM(model_id, model, tokenizer, max_new_tokens=max_new_tokens)
        old_max_new_tokens = _set_llmmap_max_new_tokens(max_new_tokens)
        try:
            entries = make_dataset_entries_for_new_llm(
                llm,
                llmmap.queries,
                prompt_confs,
                pool=TRAIN,
            )
        finally:
            _set_llmmap_max_new_tokens(old_max_new_tokens)
        candidate_template = llmmap.compute_template(entries)
        distances = llmmap.distance_fn and _llmmap_distances(
            candidate_template,
            llmmap.DB,
            llmmap.distance_fn,
        )
        nearest = nearest_llmmap_labels(distances, llmmap.label_map, top_k)
        results[model_id] = {
            "matched_reference_top1": nearest[0][0] == reference_model,
            "reference_model": reference_model,
            "top_k": [{"label": label, "distance": distance} for label, distance in nearest],
        }
        del model
    return results


def _load_llmmap_templates_into_model(llmmap, templates_path: Path) -> None:
    import numpy as np

    with templates_path.open() as f:
        templates = {key: np.array(value) for key, value in json.load(f).items()}
    llmmap.templates_map = templates
    llmmap.llms_supported = sorted(templates.keys())
    llmmap.label_map = {i: llm for i, llm in enumerate(llmmap.llms_supported)}
    llmmap.DB = np.concatenate([templates[llm][np.newaxis, :] for llm in llmmap.llms_supported])
    llmmap.ready = True


class LocalHFLLM:
    def __init__(self, llm_name: str, model, tokenizer, max_new_tokens: int):
        self.llm_name = llm_name
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.is_hf = True

    def make_prompt(self, system: str | None, user: str) -> str:
        messages = []
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            if system and "system" in chat_template:
                messages.append({"role": "system", "content": system})
            elif system:
                user = f"{system}\n\n{user}"
            messages.append({"role": "user", "content": user})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if system:
            return f"{system}\n\n{user}"
        return user

    def generate(self, prompt: str, gen_kargs: dict[str, Any], max_new_tokens: int | None = None):
        import torch

        inputs = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
        ).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **gen_kargs,
            )
        generated = [
            output_ids[i, inputs.input_ids[i].shape[0] :] for i in range(output_ids.shape[0])
        ]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


def _set_llmmap_max_new_tokens(value: int) -> int:
    import LLMmap.llm as llm_module

    old_value = llm_module.max_new_tokens
    llm_module.max_new_tokens = value
    return old_value


def _llmmap_distances(candidate_template, db, distance_fn: str):
    import numpy as np
    from scipy.spatial.distance import cdist

    return cdist(candidate_template[np.newaxis, :], db, metric=distance_fn)[0]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def run_replay_verification(args: argparse.Namespace) -> dict[str, Any]:
    proflingo_cases = load_proflingo_cases(
        args.proflingo_fingerprint,
        args.proflingo_questions,
        limit=args.limit,
    )
    trap_cases = load_trap_cases(args.trap_suffixes, limit=args.limit)

    summary: dict[str, Any] = {"proflingo": {}, "trap": {}}
    for model_id in args.models:
        model, tokenizer = load_hf_model(model_id, dtype=args.dtype, device_map=args.device_map)
        proflingo_result = evaluate_replay_cases(
            "proflingo",
            model_id,
            proflingo_cases,
            model,
            tokenizer,
            args.max_new_tokens,
            args.proflingo_match,
        )
        trap_result = evaluate_replay_cases(
            "trap",
            model_id,
            trap_cases,
            model,
            tokenizer,
            args.max_new_tokens,
        )
        summary["proflingo"][model_id] = asdict(proflingo_result)
        summary["trap"][model_id] = asdict(trap_result)
        del model
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify OLMo2-1B-Instruct fingerprints by replaying reference probes "
            "against OLMo2 base and instruct models."
        )
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--reference-model", default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "artifacts/verification/olmo2_fingerprint_verification.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-adversarial", action="store_true")
    parser.add_argument("--skip-llmmap", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--proflingo-match",
        choices=["exact", "prefix", "contains"],
        default="prefix",
        help=(
            "Automated proxy for ProFLingo's target-at-first-place success criterion. "
            "Rows always include exact/prefix/contains diagnostics."
        ),
    )
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--proflingo-fingerprint",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "proflingo/generated_olmo2_0425_1b_instruct.txt",
    )
    parser.add_argument("--proflingo-questions", type=Path, default=DEFAULT_PROFLINGO_QUESTIONS_PATH)
    parser.add_argument(
        "--trap-suffixes",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "trap/suffixes.csv",
    )
    parser.add_argument("--llmmap-model-path", type=Path, default=DEFAULT_LLMMAP_MODEL_PATH)
    parser.add_argument(
        "--llmmap-templates",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "llmmap/templates.json",
    )
    parser.add_argument("--llmmap-prompt-conf-path", type=Path, default=DEFAULT_LLMMAP_PROMPT_CONF_PATH)
    parser.add_argument("--llmmap-num-prompt-confs", type=int, default=10)
    parser.add_argument("--llmmap-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    return parser


def resolve_artifact_defaults(args: argparse.Namespace) -> None:
    if args.artifact_root == DEFAULT_ARTIFACT_ROOT:
        return
    default_proflingo = DEFAULT_ARTIFACT_ROOT / "proflingo/generated_olmo2_0425_1b_instruct.txt"
    default_trap = DEFAULT_ARTIFACT_ROOT / "trap/suffixes.csv"
    default_llmmap_templates = DEFAULT_ARTIFACT_ROOT / "llmmap/templates.json"
    if args.proflingo_fingerprint == default_proflingo:
        args.proflingo_fingerprint = args.artifact_root / "proflingo/generated_olmo2_0425_1b_instruct.txt"
    if args.trap_suffixes == default_trap:
        args.trap_suffixes = args.artifact_root / "trap/suffixes.csv"
    if args.llmmap_templates == default_llmmap_templates:
        args.llmmap_templates = args.artifact_root / "llmmap/templates.json"


def main() -> int:
    args = build_parser().parse_args()
    resolve_artifact_defaults(args)
    report: dict[str, Any] = {
        "reference_model": args.reference_model,
        "models_of_interest": args.models,
    }

    if not args.skip_adversarial:
        report.update(run_replay_verification(args))

    if not args.skip_llmmap:
        report["llmmap"] = run_llmmap_verification(
            args.models,
            args.reference_model,
            args.llmmap_model_path,
            args.llmmap_templates,
            args.llmmap_prompt_conf_path,
            args.llmmap_num_prompt_confs,
            args.llmmap_top_k,
            args.max_new_tokens,
            args.seed,
            args.dtype,
            args.device_map,
        )

    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"\nWrote verification report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
