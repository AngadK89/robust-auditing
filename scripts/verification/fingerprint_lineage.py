"""Reusable model-lineage configuration and target expansion helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


RevisionLister = Callable[[str], list[str]]


@dataclass(frozen=True)
class FingerprintReference:
    model_id: str
    artifact_root: Path
    artifacts: dict[str, Path] = field(default_factory=dict)

    def to_report_dict(self) -> dict[str, str]:
        report = {
            "model_id": self.model_id,
            "artifact_root": str(self.artifact_root),
        }
        if self.artifacts:
            report["artifacts"] = {
                name: str(path) for name, path in sorted(self.artifacts.items())
            }
        return report


@dataclass(frozen=True)
class DiscoverRevisions:
    pattern: str
    increment: int


@dataclass(frozen=True)
class TargetSpec:
    label: str
    model_id: str
    revisions: list[str | dict[str, Any]] = field(default_factory=list)
    discover_revisions: DiscoverRevisions | None = None


@dataclass(frozen=True)
class ModelTarget:
    label: str
    model_id: str
    revision: str = "main"
    step: int | None = None

    @property
    def key(self) -> str:
        return f"{self.label}@{self.revision or 'main'}"

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model_id": self.model_id,
            "revision": self.revision or "main",
            "step": self.step,
        }


@dataclass
class LineageConfig:
    name: str
    reference: FingerprintReference
    targets: list[TargetSpec]
    output: Path | None = None
    fingerprints: dict[str, dict[str, Any]] = field(default_factory=dict)
    discovery_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_lineage_config(path: Path | str) -> LineageConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
        raise RuntimeError("PyYAML is required to load fingerprint lineage YAML files") from exc

    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Lineage config must be a mapping: {path}")
    name = _required_string(raw, "name", "lineage config")
    reference_raw = _required_mapping(raw, "reference", "lineage config")
    reference = FingerprintReference(
        model_id=_required_string(reference_raw, "model_id", "reference"),
        artifact_root=Path(_required_string(reference_raw, "artifact_root", "reference")),
        artifacts=_parse_artifact_paths(reference_raw.get("artifacts") or {}),
    )
    target_items = raw.get("targets")
    if not isinstance(target_items, list) or not target_items:
        raise ValueError("Lineage config must define at least one target")

    targets = [_parse_target_spec(item, index) for index, item in enumerate(target_items)]
    output = Path(raw["output"]) if raw.get("output") else None
    fingerprints = _parse_fingerprint_options(raw.get("fingerprints", {}))
    return LineageConfig(
        name=name,
        reference=reference,
        targets=targets,
        output=output,
        fingerprints=fingerprints,
    )


def _parse_target_spec(raw: Any, index: int) -> TargetSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"Target at index {index} must be a mapping")
    revisions_raw = _parse_revisions(raw.get("revisions") or [], index)
    discover_raw = raw.get("discover_revisions")
    discover = None
    if discover_raw is not None:
        if not isinstance(discover_raw, dict):
            raise ValueError(f"Target {index} discover_revisions must be a mapping")
        discover = DiscoverRevisions(
            pattern=_required_string(discover_raw, "pattern", f"target {index} discovery"),
            increment=int(discover_raw.get("increment") or 1),
        )
        if discover.increment < 1:
            raise ValueError(f"Target {index} discover_revisions.increment must be >= 1")
    return TargetSpec(
        label=_required_string(raw, "label", f"target {index}"),
        model_id=_required_string(raw, "model_id", f"target {index}"),
        revisions=revisions_raw,
        discover_revisions=discover,
    )


def _parse_artifact_paths(raw: Any) -> dict[str, Path]:
    if not isinstance(raw, dict):
        raise ValueError("reference.artifacts must be a mapping")
    artifact_paths: dict[str, Path] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise ValueError("reference.artifacts must map strings to non-empty paths")
        artifact_paths[key] = Path(value)
    return artifact_paths


def _parse_fingerprint_options(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("fingerprints must be a mapping")
    path_keys = {
        ("proflingo", "questions"),
        ("llmmap", "model_path"),
        ("llmmap", "prompt_conf_path"),
    }
    fingerprints: dict[str, dict[str, Any]] = {}
    for fingerprint, options in raw.items():
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("fingerprints must map non-empty names to option mappings")
        if not isinstance(options, dict):
            raise ValueError(f"fingerprints.{fingerprint} must be a mapping")
        parsed_options: dict[str, Any] = {}
        for key, value in options.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"fingerprints.{fingerprint} option names must be non-empty strings")
            parsed_options[key] = Path(value) if (fingerprint, key) in path_keys else value
        fingerprints[fingerprint] = parsed_options
    return fingerprints


def _parse_revisions(raw: Any, index: int) -> list[str | dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError(f"Target {index} revisions must be a list")
    revisions: list[str | dict[str, Any]] = []
    for revision in raw:
        if isinstance(revision, str):
            revisions.append(revision)
            continue
        if isinstance(revision, dict):
            name = revision.get("name")
            step = revision.get("step")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Target {index} structured revisions require a name")
            if step is not None and not isinstance(step, int):
                raise ValueError(f"Target {index} structured revision step must be an int")
            revisions.append({"name": name, "step": step})
            continue
        raise ValueError(f"Target {index} revisions must contain strings or mappings")
    return revisions


def _required_mapping(raw: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must define mapping field {key!r}")
    return value


def _required_string(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must define non-empty string field {key!r}")
    return value


def parse_step_revision(revision: str, pattern: str) -> int | None:
    match = re.fullmatch(pattern, revision)
    if not match:
        return None
    if not match.groups():
        raise ValueError(f"Revision discovery pattern must capture a numeric step: {pattern}")
    return int(match.group(1))


def expand_lineage_targets(
    config: LineageConfig,
    list_revisions: RevisionLister | None = None,
) -> list[ModelTarget]:
    list_revisions = list_revisions or list_hf_revisions
    targets: list[ModelTarget] = []
    for spec in config.targets:
        revisions = _explicit_revisions(spec)
        if spec.discover_revisions is not None:
            discovered = _discover_target_revisions(spec, spec.discover_revisions, list_revisions)
            config.discovery_metadata[spec.label] = discovered["metadata"]
            revisions.extend(discovered["revisions"])
        for revision, step in _dedupe_revisions(revisions):
            targets.append(
                ModelTarget(
                    label=spec.label,
                    model_id=spec.model_id,
                    revision=revision,
                    step=step,
                )
            )
    return targets


def _explicit_revisions(spec: TargetSpec) -> list[tuple[str, int | None]]:
    revisions = spec.revisions or ["main"]
    parsed: list[tuple[str, int | None]] = []
    for revision in revisions:
        if isinstance(revision, dict):
            parsed.append((str(revision["name"]), revision.get("step")))
            continue
        step = (
            parse_step_revision(revision, spec.discover_revisions.pattern)
            if revision != "main" and spec.discover_revisions is not None
            else None
        )
        parsed.append((revision, step))
    return parsed


def _discover_target_revisions(
    spec: TargetSpec,
    discovery: DiscoverRevisions,
    list_revisions: RevisionLister,
) -> dict[str, Any]:
    available = list_revisions(spec.model_id)
    matched: list[tuple[str, int]] = []
    skipped: list[str] = []
    for revision in available:
        step = parse_step_revision(revision, discovery.pattern)
        if step is None:
            continue
        if step % discovery.increment == 0:
            matched.append((revision, step))
        else:
            skipped.append(revision)
    matched.sort(key=lambda item: item[1])
    return {
        "revisions": [(revision, step) for revision, step in matched],
        "metadata": {
            "pattern": discovery.pattern,
            "increment": discovery.increment,
            "available_revisions": available,
            "matched_revisions": [revision for revision, _step in matched],
            "skipped_revisions": skipped,
        },
    }


def _dedupe_revisions(revisions: list[tuple[str, int | None]]) -> list[tuple[str, int | None]]:
    seen: set[str] = set()
    deduped: list[tuple[str, int | None]] = []
    for revision, step in revisions:
        normalized = revision or "main"
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((normalized, step))
    return deduped


def list_hf_revisions(model_id: str) -> list[str]:
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(model_id, repo_type="model")
    branches = [branch.name for branch in refs.branches]
    tags = [tag.name for tag in refs.tags]
    return branches + [tag for tag in tags if tag not in branches]
