"""Shared fingerprint loading, replay, and LLMmap verification methods."""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LLMMAP_MODEL_PATH = ROOT_DIR / "third_party/LLMmap/data/pretrained_models/default"
DEFAULT_LLMMAP_PROMPT_CONF_PATH = ROOT_DIR / "third_party/LLMmap/confs/prompt_configurations"
DEFAULT_PROFLINGO_QUESTIONS_PATH = ROOT_DIR / "third_party/ProFLingo/questions.csv"
ALL_FINGERPRINTS = ("proflingo", "trap", "llmmap")


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


def load_hf_model(
    model_id: str,
    dtype: str = "auto",
    device_map: str = "auto",
    revision: str | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
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
        revision=revision,
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
    model_id: str,
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
    revision: str | None = None,
    result_key: str | None = None,
    target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = result_key or model_id
    model, tokenizer = load_hf_model(
        model_id,
        dtype=dtype,
        device_map=device_map,
        revision=revision,
    )
    try:
        result = run_llmmap_verification_for_loaded_model(
            model_id,
            reference_model,
            llmmap_model_path,
            llmmap_templates_path,
            prompt_conf_path,
            num_prompt_confs,
            top_k,
            max_new_tokens,
            seed,
            model,
            tokenizer,
            result_key=key,
            target_metadata=target_metadata,
        )
    finally:
        del model
        del tokenizer
    results: dict[str, Any] = {}
    results[key] = result
    return results


def run_llmmap_verification_for_loaded_model(
    model_id: str,
    reference_model: str,
    llmmap_model_path: Path,
    llmmap_templates_path: Path,
    prompt_conf_path: Path,
    num_prompt_confs: int,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    model,
    tokenizer,
    result_key: str | None = None,
    target_metadata: dict[str, Any] | None = None,
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
            "Build or provide an LLMmap template artifact for the reference model first."
        )

    random.seed(seed)
    prompt_confs = PromptConfFactory(prompt_conf_path).sample(num_prompt_confs, pool=TRAIN)
    key = result_key or model_id
    llm = LocalHFLLM(key, model, tokenizer, max_new_tokens=max_new_tokens)
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
    result = {
        "matched_reference_top1": nearest[0][0] == reference_model,
        "reference_model": reference_model,
        "top_k": [{"label": label, "distance": distance} for label, distance in nearest],
    }
    if target_metadata is not None:
        result["target"] = target_metadata
    return result


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


def require_configured_path(path: Path | None, description: str) -> Path:
    if path is None:
        raise ValueError(f"Missing configured {description}")
    return path


def cleanup_torch_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolved_hf_revision(model, tokenizer, requested_revision: str | None = None) -> str:
    for candidate in (model, tokenizer):
        config = getattr(candidate, "config", None)
        commit_hash = getattr(config, "_commit_hash", None)
        if commit_hash:
            return str(commit_hash)
        init_kwargs = getattr(candidate, "init_kwargs", None)
        if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
            return str(init_kwargs["_commit_hash"])
    return requested_revision or "main"


def evict_hf_model_cache(model_id: str, revision: str | None = None) -> bool:
    revision = revision or "main"
    try:
        cache_info = scan_cache_dir()
    except Exception as exc:
        print(
            f"Warning: could not scan Hugging Face cache while evicting "
            f"{model_id}@{revision}: {exc}",
            file=sys.stderr,
        )
        return False

    for warning in getattr(cache_info, "warnings", []) or []:
        print(f"Warning from Hugging Face cache scan: {warning}", file=sys.stderr)

    target_repo = None
    for repo in getattr(cache_info, "repos", []) or []:
        if getattr(repo, "repo_id", None) == model_id and getattr(repo, "repo_type", "model") in {
            None,
            "model",
        }:
            target_repo = repo
            break
    if target_repo is None:
        print(
            f"Warning: Could not find Hugging Face cache repo for {model_id}; "
            "no cache entry deleted.",
            file=sys.stderr,
        )
        return False

    revision_hash = None
    for cached_revision in getattr(target_repo, "revisions", []) or []:
        commit_hash = getattr(cached_revision, "commit_hash", None)
        refs = set(getattr(cached_revision, "refs", []) or [])
        if revision == commit_hash or revision in refs:
            revision_hash = commit_hash
            break
    if revision_hash is None:
        print(
            f"Warning: Could not find Hugging Face cache revision {model_id}@{revision}; "
            "no cache entry deleted.",
            file=sys.stderr,
        )
        return False

    try:
        delete_strategy = cache_info.delete_revisions(revision_hash)
        expected = getattr(delete_strategy, "expected_freed_size_str", "unknown size")
        cache_dir = getattr(cache_info, "cache_dir", None) or "active Hugging Face cache"
        print(
            f"Deleting Hugging Face cache for {model_id}@{revision} "
            f"from {cache_dir}; expected to free {expected}.",
            file=sys.stderr,
        )
        delete_strategy.execute()
        return True
    except Exception as exc:
        print(
            f"Warning: failed to delete Hugging Face cache for {model_id}@{revision}: {exc}",
            file=sys.stderr,
        )
        return False


def _hf_model_cache_folder_name(model_id: str) -> str | None:
    model_id = model_id.strip()
    if not model_id or "\0" in model_id or "\\" in model_id:
        return None
    return "models--" + model_id.replace("/", "--")


def _safe_child_path(root: Path, child_name: str) -> Path | None:
    root = root.expanduser().resolve(strict=False)
    child = (root / child_name).resolve(strict=False)
    if child == root or not child.is_relative_to(root):
        return None
    return child


def _remove_cache_path(path: Path, description: str, *, require_directory: bool = False) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink():
        print(
            f"Warning: refusing to delete symlinked Hugging Face cache {description}: {path}",
            file=sys.stderr,
        )
        return False
    if path.is_dir():
        shutil.rmtree(path)
        return True
    if require_directory:
        print(
            f"Warning: refusing to delete non-directory Hugging Face cache {description}: {path}",
            file=sys.stderr,
        )
        return False
    path.unlink()
    return True


def evict_hf_repo_cache(model_id: str, cache_dir: Path | str | None = None) -> bool:
    """Remove the whole cached Hugging Face model repo for a model id.

    This is intentionally repo-scoped, not revision-scoped: Transformers may cache
    auxiliary revisions such as refs/pr/1 or leave stale refs that make
    scan_cache_dir() unable to clean later revisions.
    """
    folder_name = _hf_model_cache_folder_name(model_id)
    if folder_name is None:
        print(
            f"Warning: refusing to delete Hugging Face cache for invalid model id: {model_id!r}",
            file=sys.stderr,
        )
        return False

    cache_root = Path(cache_dir) if cache_dir is not None else Path(HF_HUB_CACHE)
    repo_path = _safe_child_path(cache_root, folder_name)
    lock_path = _safe_child_path(cache_root / ".locks", folder_name)
    if repo_path is None or lock_path is None:
        print(
            f"Warning: refusing to delete Hugging Face cache for {model_id}; "
            "computed path escaped the cache root.",
            file=sys.stderr,
        )
        return False

    removed_repo = _remove_cache_path(
        repo_path,
        f"repo for {model_id}",
        require_directory=True,
    )
    removed_lock = _remove_cache_path(lock_path, f"locks for {model_id}")
    if removed_repo or removed_lock:
        print(
            f"Deleted Hugging Face cache repo for {model_id} from {cache_root}.",
            file=sys.stderr,
        )
        return True

    print(
        f"Warning: Could not find Hugging Face cache repo folder for {model_id}; "
        "no cache entry deleted.",
        file=sys.stderr,
    )
    return False
