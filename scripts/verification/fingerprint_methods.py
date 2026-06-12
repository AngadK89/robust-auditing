"""Shared fingerprint loading and LLMmap verification methods."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LLMMAP_MODEL_PATH = ROOT_DIR / "third_party/LLMmap/data/pretrained_models/default"
DEFAULT_PROFLINGO_QUESTIONS_PATH = ROOT_DIR / "third_party/ProFLingo/questions.csv"
ALL_FINGERPRINTS = ("proflingo", "llmmap")


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_hf_model(
    model_id: str,
    dtype: str = "auto",
    device_map: str = "auto",
    revision: str | None = None,
    use_fast: bool = True,
):
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_hf_tokenizer(
        model_id,
        revision=revision,
        use_fast=use_fast,
    )

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


def load_hf_tokenizer(
    model_id: str,
    revision: str | None = None,
    use_fast: bool = True,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        use_fast=use_fast,
        token=os.environ.get("HUGGINGFACE_API_KEY") or None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


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
    top_k: int,
    max_new_tokens: int,
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
            top_k,
            max_new_tokens,
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
    top_k: int,
    max_new_tokens: int,
    model,
    tokenizer,
    result_key: str | None = None,
    target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT_DIR / "third_party/LLMmap"))
    from LLMmap.inference import load_LLMmap

    require_path(llmmap_model_path, "LLMmap pretrained model directory")
    require_path(llmmap_templates_path, "LLMmap template artifact")

    _conf, llmmap = load_LLMmap(str(llmmap_model_path), device="cpu", verbose=False)
    _load_llmmap_templates_into_model(llmmap, llmmap_templates_path)
    if reference_model not in llmmap.templates_map:
        raise ValueError(
            f"Reference model {reference_model!r} is not in LLMmap templates. "
            "Build or provide an LLMmap template artifact for the reference model first."
        )

    key = result_key or model_id
    llm = LocalHFLLM(key, model, tokenizer, max_new_tokens=max_new_tokens)
    queries = list(llmmap.queries)
    answers: list[str] = []
    traces: list[dict[str, str]] = []
    for query in queries:
        prompt = llm.make_prompt(None, query)
        response = llm.generate(prompt, {})[0]
        answers.append(response)
        traces.append({"query": query, "response": response})

    distances = llmmap(answers)
    nearest = nearest_llmmap_labels(distances, llmmap.label_map, top_k)
    result = {
        "matched_reference_top1": nearest[0][0] == reference_model,
        "reference_model": reference_model,
        "top_k": [{"label": label, "distance": distance} for label, distance in nearest],
        "verification_mode": "direct_queries",
        "query_count": len(queries),
        "template_count": len(llmmap.templates_map),
        "distance_fn": llmmap.distance_fn,
        "traces": traces,
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
    llmmap.distance_fn = getattr(llmmap, "conf", {}).get("distance_fn", "euclidean")
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
