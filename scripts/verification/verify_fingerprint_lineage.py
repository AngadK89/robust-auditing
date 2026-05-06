#!/usr/bin/env python3
"""Verify reference fingerprints across a configured model lineage."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from scripts.verification.fingerprint_lineage import (
    LineageConfig,
    ModelTarget,
    expand_lineage_targets,
    load_lineage_config,
)
from scripts.verification.fingerprint_methods import (
    ALL_FINGERPRINTS,
    DEFAULT_LLMMAP_MODEL_PATH,
    DEFAULT_LLMMAP_PROMPT_CONF_PATH,
    DEFAULT_PROFLINGO_QUESTIONS_PATH,
    ROOT_DIR,
    cleanup_torch_memory,
    evaluate_replay_cases,
    evict_hf_repo_cache,
    load_hf_model,
    load_proflingo_cases,
    load_trap_cases,
    resolved_hf_revision,
    run_llmmap_verification_for_loaded_model,
    write_json,
)


DEFAULT_PROFLINGO_MATCH = "prefix"
DEFAULT_LLMMAP_NUM_PROMPT_CONFS = 10
DEFAULT_LLMMAP_TOP_K = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify reference fingerprints across a YAML-configured model lineage."
    )
    parser.add_argument("--lineage-config", type=Path, required=True)
    parser.add_argument(
        "--fingerprint",
        nargs="+",
        choices=ALL_FINGERPRINTS,
        default=None,
        help="Fingerprint technique(s) to verify. Defaults to techniques declared in the lineage YAML artifacts.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=41)
    return parser


@dataclass(frozen=True)
class ProflingoOptions:
    questions: Path = DEFAULT_PROFLINGO_QUESTIONS_PATH
    match: str = DEFAULT_PROFLINGO_MATCH


@dataclass(frozen=True)
class LLMmapOptions:
    model_path: Path = DEFAULT_LLMMAP_MODEL_PATH
    prompt_conf_path: Path = DEFAULT_LLMMAP_PROMPT_CONF_PATH
    num_prompt_confs: int = DEFAULT_LLMMAP_NUM_PROMPT_CONFS
    top_k: int = DEFAULT_LLMMAP_TOP_K


REQUIRED_ARTIFACTS_BY_FINGERPRINT = {
    "proflingo": "proflingo_fingerprint",
    "trap": "trap_suffixes",
    "llmmap": "llmmap_templates",
}


def configured_fingerprints(config: LineageConfig) -> list[str]:
    configured_artifacts = set(config.reference.artifacts)
    fingerprints = [
        fingerprint
        for fingerprint in ALL_FINGERPRINTS
        if REQUIRED_ARTIFACTS_BY_FINGERPRINT[fingerprint] in configured_artifacts
    ]
    if not fingerprints:
        raise ValueError(
            "Lineage reference.artifacts does not declare any supported fingerprint artifacts. "
            "Add one of: " + ", ".join(sorted(REQUIRED_ARTIFACTS_BY_FINGERPRINT.values()))
        )
    return fingerprints


def artifact_paths(config: LineageConfig, fingerprints: list[str]) -> dict[str, Path]:
    root = config.reference.artifact_root
    configured = config.reference.artifacts
    required = {
        REQUIRED_ARTIFACTS_BY_FINGERPRINT[fingerprint]
        for fingerprint in fingerprints
        if fingerprint in REQUIRED_ARTIFACTS_BY_FINGERPRINT
    }
    missing = sorted(required - set(configured))
    if missing:
        raise ValueError(
            "Lineage reference.artifacts is missing required path(s): "
            + ", ".join(missing)
        )
    return {name: root / path for name, path in configured.items()}


def proflingo_options(config: LineageConfig) -> ProflingoOptions:
    options = config.fingerprints.get("proflingo", {})
    match = str(options.get("match", DEFAULT_PROFLINGO_MATCH))
    if match not in {"exact", "prefix", "contains"}:
        raise ValueError("fingerprints.proflingo.match must be one of: exact, prefix, contains")
    return ProflingoOptions(
        questions=Path(options.get("questions", DEFAULT_PROFLINGO_QUESTIONS_PATH)),
        match=match,
    )


def llmmap_options(config: LineageConfig) -> LLMmapOptions:
    options = config.fingerprints.get("llmmap", {})
    num_prompt_confs = int(options.get("num_prompt_confs", DEFAULT_LLMMAP_NUM_PROMPT_CONFS))
    top_k = int(options.get("top_k", DEFAULT_LLMMAP_TOP_K))
    if num_prompt_confs < 1:
        raise ValueError("fingerprints.llmmap.num_prompt_confs must be >= 1")
    if top_k < 1:
        raise ValueError("fingerprints.llmmap.top_k must be >= 1")
    return LLMmapOptions(
        model_path=Path(options.get("model_path", DEFAULT_LLMMAP_MODEL_PATH)),
        prompt_conf_path=Path(options.get("prompt_conf_path", DEFAULT_LLMMAP_PROMPT_CONF_PATH)),
        num_prompt_confs=num_prompt_confs,
        top_k=top_k,
    )


def default_output_path(config: LineageConfig) -> Path:
    return config.output or ROOT_DIR / "artifacts/verification" / f"{config.name}.json"


def run_proflingo_for_target(
    *,
    target: ModelTarget,
    cases,
    model,
    tokenizer,
    max_new_tokens: int,
    proflingo_match: str,
) -> dict[str, Any]:
    result = asdict(
        evaluate_replay_cases(
            "proflingo",
            target.key,
            cases,
            model,
            tokenizer,
            max_new_tokens,
            proflingo_match,
        )
    )
    result["target"] = target.to_report_dict()
    return result


def run_trap_for_target(
    *,
    target: ModelTarget,
    cases,
    model,
    tokenizer,
    max_new_tokens: int,
) -> dict[str, Any]:
    result = asdict(
        evaluate_replay_cases(
            "trap",
            target.key,
            cases,
            model,
            tokenizer,
            max_new_tokens,
        )
    )
    result["target"] = target.to_report_dict()
    return result


def run_llmmap_for_target(
    *,
    args: argparse.Namespace,
    config: LineageConfig,
    options: LLMmapOptions | None = None,
    target: ModelTarget,
    model,
    tokenizer,
) -> dict[str, Any]:
    options = options or llmmap_options(config)
    return run_llmmap_verification_for_loaded_model(
        target.model_id,
        config.reference.model_id,
        options.model_path,
        args.llmmap_templates,
        options.prompt_conf_path,
        options.num_prompt_confs,
        options.top_k,
        args.max_new_tokens,
        args.seed,
        model,
        tokenizer,
        result_key=target.key,
        target_metadata=target.to_report_dict(),
    )


def cleanup_after_target_model(model_id: str, revision: str | None) -> None:
    cleanup_torch_memory()
    evict_hf_repo_cache(model_id)


def build_report(
    config: LineageConfig,
    targets: list[ModelTarget],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "lineage": config.name,
        "reference": config.reference.to_report_dict(),
        "reference_model": config.reference.model_id,
        "artifact_root": str(config.reference.artifact_root),
        "fingerprint": args.fingerprint,
        "targets": [target.key for target in targets],
        "target_metadata": {target.key: target.to_report_dict() for target in targets},
        "discovery": config.discovery_metadata,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_lineage_config(args.lineage_config)
    targets = expand_lineage_targets(config)
    args.fingerprint = args.fingerprint or configured_fingerprints(config)
    paths = artifact_paths(config, args.fingerprint)
    args.proflingo_fingerprint = paths.get("proflingo_fingerprint")
    args.trap_suffixes = paths.get("trap_suffixes")
    args.llmmap_templates = paths.get("llmmap_templates")
    output = args.output or default_output_path(config)

    requested = set(args.fingerprint)
    proflingo_config = proflingo_options(config) if "proflingo" in requested else None
    llmmap_config = llmmap_options(config) if "llmmap" in requested else None
    proflingo_cases = (
        load_proflingo_cases(args.proflingo_fingerprint, proflingo_config.questions, limit=args.limit)
        if "proflingo" in requested
        else []
    )
    trap_cases = (
        load_trap_cases(args.trap_suffixes, limit=args.limit)
        if "trap" in requested
        else []
    )

    report = build_report(config, targets, args)
    for fingerprint in args.fingerprint:
        report[fingerprint] = {}

    for target in targets:
        model = None
        tokenizer = None
        cleanup_revision = target.revision
        try:
            model, tokenizer = load_hf_model(
                target.model_id,
                dtype=args.dtype,
                device_map=args.device_map,
                revision=target.revision,
            )
            cleanup_revision = resolved_hf_revision(model, tokenizer, target.revision)
            if "proflingo" in requested:
                report["proflingo"][target.key] = run_proflingo_for_target(
                    target=target,
                    cases=proflingo_cases,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    proflingo_match=proflingo_config.match,
                )
            if "trap" in requested:
                report["trap"][target.key] = run_trap_for_target(
                    target=target,
                    cases=trap_cases,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                )
            if "llmmap" in requested:
                report["llmmap"][target.key] = run_llmmap_for_target(
                    args=args,
                    config=config,
                    options=llmmap_config,
                    target=target,
                    model=model,
                    tokenizer=tokenizer,
                )
        finally:
            del model
            del tokenizer
            cleanup_after_target_model(target.model_id, cleanup_revision)

    write_json(output, report)
    print(json.dumps(report, indent=2))
    print(f"\nWrote lineage verification report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
