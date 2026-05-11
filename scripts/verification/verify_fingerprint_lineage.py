#!/usr/bin/env python3
"""Verify reference fingerprints across a configured model lineage."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
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
    DEFAULT_PROFLINGO_QUESTIONS_PATH,
    ROOT_DIR,
    cleanup_torch_memory,
    evaluate_replay_cases,
    evict_hf_repo_cache,
    load_hf_model,
    load_hf_tokenizer,
    load_trap_cases,
    require_path,
    resolved_hf_revision,
    run_llmmap_verification_for_loaded_model,
    write_json,
)


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
    return parser


@dataclass(frozen=True)
class ProflingoOptions:
    questions: Path = DEFAULT_PROFLINGO_QUESTIONS_PATH


@dataclass(frozen=True)
class LLMmapOptions:
    model_path: Path = DEFAULT_LLMMAP_MODEL_PATH
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
    allowed_keys = {"questions"}
    unknown_keys = sorted(set(options) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            "fingerprints.proflingo contains unsupported option(s): "
            + ", ".join(unknown_keys)
        )
    return ProflingoOptions(
        questions=Path(options.get("questions", DEFAULT_PROFLINGO_QUESTIONS_PATH)),
    )


def llmmap_options(config: LineageConfig) -> LLMmapOptions:
    options = config.fingerprints.get("llmmap", {})
    allowed_keys = {"model_path", "top_k"}
    unknown_keys = sorted(set(options) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            "fingerprints.llmmap contains unsupported option(s): "
            + ", ".join(unknown_keys)
        )
    top_k = int(options.get("top_k", DEFAULT_LLMMAP_TOP_K))
    if top_k < 1:
        raise ValueError("fingerprints.llmmap.top_k must be >= 1")
    return LLMmapOptions(
        model_path=Path(options.get("model_path", DEFAULT_LLMMAP_MODEL_PATH)),
        top_k=top_k,
    )


def default_output_path(config: LineageConfig) -> Path:
    return config.output or ROOT_DIR / "artifacts/verification" / f"{config.name}.json"


def run_proflingo_for_target(
    *,
    target: ModelTarget,
    fingerprint_path: Path,
    questions_path: Path,
    model,
    tokenizer,
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    require_path(fingerprint_path, "ProFLingo fingerprint file")
    require_path(questions_path, "ProFLingo questions CSV")
    with _limited_proflingo_fingerprint(fingerprint_path, limit) as advsamples_path:
        total, matched = run_proflingo_copyright_test(
            model=model,
            tokenizer=tokenizer,
            dataset_path=questions_path,
            advsamples_path=advsamples_path,
            manual_check=False,
            model_path=target.model_id,
            template=get_proflingo_default_templates(target.model_id),
            verbose=False,
            max_token=max_new_tokens,
        )
    return {
        "technique": "proflingo",
        "model": target.key,
        "total": total,
        "matched": matched,
        "match_rate": (matched / total) if total else 0.0,
        "target": target.to_report_dict(),
        "verification_mode": "proflingo_copyright_test",
    }


def _load_proflingo_modules():
    proflingo_dir = ROOT_DIR / "third_party/ProFLingo"
    if str(proflingo_dir) not in sys.path:
        sys.path.insert(0, str(proflingo_dir))
    import copyright_test
    import proflingo

    return copyright_test, proflingo


def get_proflingo_default_templates(model_id: str):
    _copyright_test, proflingo = _load_proflingo_modules()
    try:
        return _copyright_test.get_template(model_id)
    except ValueError:
        template = proflingo.get_conv_template("zero_shot")
        template.sep = "\n"
        return template


def run_proflingo_copyright_test(**kwargs):
    copyright_test, _proflingo = _load_proflingo_modules()
    return copyright_test.fingerprint_test(**kwargs)


class _limited_proflingo_fingerprint:
    def __init__(self, fingerprint_path: Path, limit: int | None):
        self.fingerprint_path = fingerprint_path
        self.limit = limit
        self._temporary_file = None

    def __enter__(self) -> Path:
        if self.limit is None:
            return self.fingerprint_path
        self._temporary_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".txt",
            delete=False,
        )
        with self.fingerprint_path.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                if index >= self.limit:
                    break
                self._temporary_file.write(line)
        self._temporary_file.close()
        return Path(self._temporary_file.name)

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._temporary_file is not None:
            Path(self._temporary_file.name).unlink(missing_ok=True)


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
        options.top_k,
        args.max_new_tokens,
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
        proflingo_tokenizer = None
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
                proflingo_tokenizer = load_hf_tokenizer(
                    target.model_id,
                    revision=target.revision,
                    use_fast=False,
                )
                report["proflingo"][target.key] = run_proflingo_for_target(
                    target=target,
                    fingerprint_path=args.proflingo_fingerprint,
                    questions_path=proflingo_config.questions,
                    model=model,
                    tokenizer=proflingo_tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    limit=args.limit,
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
            del proflingo_tokenizer
            cleanup_after_target_model(target.model_id, cleanup_revision)

    write_json(output, report)
    print(json.dumps(report, indent=2))
    print(f"\nWrote lineage verification report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
