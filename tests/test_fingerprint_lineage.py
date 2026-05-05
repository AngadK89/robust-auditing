import json
from pathlib import Path

from scripts.verification.fingerprint_lineage import (
    DiscoverRevisions,
    FingerprintReference,
    LineageConfig,
    ModelTarget,
    TargetSpec,
    expand_lineage_targets,
    load_lineage_config,
    parse_step_revision,
)
from scripts.verification import verify_fingerprint_lineage


def test_load_lineage_config_resolves_reference_and_targets(tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    config_path.write_text(
        """
name: custom_family
reference:
  model_id: org/reference
  artifact_root: artifacts/fingerprints/reference
output: artifacts/verification/custom_family.json
targets:
  - label: base
    model_id: org/base
  - label: tuned
    model_id: org/tuned
    revisions:
      - main
      - checkpoint_a
  - label: rl
    model_id: org/rl
    discover_revisions:
      pattern: step_(\\d+)
      increment: 200
""".strip()
    )

    config = load_lineage_config(config_path)

    assert config == LineageConfig(
        name="custom_family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[
            TargetSpec(label="base", model_id="org/base"),
            TargetSpec(label="tuned", model_id="org/tuned", revisions=["main", "checkpoint_a"]),
            TargetSpec(
                label="rl",
                model_id="org/rl",
                discover_revisions=DiscoverRevisions(pattern=r"step_(\d+)", increment=200),
            ),
        ],
        output=Path("artifacts/verification/custom_family.json"),
    )


def test_load_lineage_config_requires_targets(tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    config_path.write_text(
        """
name: empty
reference:
  model_id: org/reference
  artifact_root: artifacts/fingerprints/reference
targets: []
""".strip()
    )

    try:
        load_lineage_config(config_path)
    except ValueError as exc:
        assert "at least one target" in str(exc)
    else:
        raise AssertionError("empty lineage targets should be rejected")


def test_parse_step_revision_uses_configured_regex():
    assert parse_step_revision("step_200", r"step_(\d+)") == 200
    assert parse_step_revision("step_0200", r"step_(\d+)") == 200
    assert parse_step_revision("checkpoint-200", r"step_(\d+)") is None


def test_explicit_revisions_support_structured_steps_and_target_discovery_pattern():
    config = LineageConfig(
        name="family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[
            TargetSpec(
                label="rl",
                model_id="org/rl",
                revisions=["checkpoint-200", {"name": "checkpoint-final", "step": 300}],
                discover_revisions=DiscoverRevisions(pattern=r"checkpoint-(\d+)", increment=100),
            ),
        ],
    )

    targets = expand_lineage_targets(config, list_revisions=lambda _model_id: [])

    assert targets == [
        ModelTarget(label="rl", model_id="org/rl", revision="checkpoint-200", step=200),
        ModelTarget(label="rl", model_id="org/rl", revision="checkpoint-final", step=300),
    ]


def test_expand_lineage_targets_adds_main_and_sorted_incremental_discovered_revisions():
    config = LineageConfig(
        name="family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[
            TargetSpec(label="base", model_id="org/base"),
            TargetSpec(
                label="rl",
                model_id="org/rl",
                discover_revisions=DiscoverRevisions(pattern=r"step_(\d+)", increment=200),
            ),
        ],
    )

    def fake_refs(model_id: str) -> list[str]:
        assert model_id == "org/rl"
        return ["main", "step_20", "step_400", "step_200", "step_abc", "other"]

    targets = expand_lineage_targets(config, list_revisions=fake_refs)

    assert targets == [
        ModelTarget(label="base", model_id="org/base", revision="main"),
        ModelTarget(label="rl", model_id="org/rl", revision="main"),
        ModelTarget(label="rl", model_id="org/rl", revision="step_200", step=200),
        ModelTarget(label="rl", model_id="org/rl", revision="step_400", step=400),
    ]
    assert [target.key for target in targets] == [
        "base@main",
        "rl@main",
        "rl@step_200",
        "rl@step_400",
    ]


def test_expand_lineage_targets_records_missing_discovery_without_dropping_main():
    config = LineageConfig(
        name="family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[
            TargetSpec(
                label="rl",
                model_id="org/rl",
                discover_revisions=DiscoverRevisions(pattern=r"step_(\d+)", increment=200),
            ),
        ],
    )

    targets = expand_lineage_targets(config, list_revisions=lambda _model_id: ["main", "step_20"])

    assert targets == [ModelTarget(label="rl", model_id="org/rl", revision="main")]
    assert config.discovery_metadata["rl"]["matched_revisions"] == []
    assert config.discovery_metadata["rl"]["available_revisions"] == ["main", "step_20"]


def test_lineage_cli_runs_model_first_and_preserves_report_shape(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    output_path = tmp_path / "report.json"
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "proflingo").mkdir(parents=True)
    (artifact_root / "llmmap").mkdir()
    (artifact_root / "proflingo/generated.txt").write_text("0,suffix\n")
    (artifact_root / "llmmap/templates.json").write_text("{}")
    config_path.write_text(
        f"""
name: smoke_family
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated.txt
    llmmap_templates: llmmap/templates.json
targets:
  - label: base
    model_id: org/base
  - label: rl
    model_id: org/rl
    revisions:
      - main
      - name: step_200
        step: 200
""".strip()
    )

    calls = []

    def fake_load(model_id, **kwargs):
        revision = kwargs.get("revision")
        calls.append(("load", model_id, revision))
        return f"model:{model_id}:{revision}", f"tokenizer:{model_id}:{revision}"

    def fake_proflingo(*, target, **_kwargs):
        calls.append(("proflingo", target.key))
        return {"match_rate": 1.0, "target": target.to_report_dict()}

    def fake_llmmap(*, target, **_kwargs):
        calls.append(("llmmap", target.key))
        return {
            "matched_reference_top1": True,
            "reference_model": "org/reference",
            "top_k": [],
            "target": target.to_report_dict(),
        }

    def fake_cleanup(model_id, revision):
        calls.append(("cleanup", model_id, revision))

    def fake_write_json(path, data):
        path.write_text(json.dumps(data, indent=2))

    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_model", fake_load)
    monkeypatch.setattr(verify_fingerprint_lineage, "load_proflingo_cases", lambda *_args, **_kwargs: ["case"])
    monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_for_target", fake_proflingo)
    monkeypatch.setattr(verify_fingerprint_lineage, "run_llmmap_for_target", fake_llmmap)
    monkeypatch.setattr(verify_fingerprint_lineage, "cleanup_after_target_model", fake_cleanup)
    monkeypatch.setattr(verify_fingerprint_lineage, "write_json", fake_write_json)
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_fingerprint_lineage.py",
            "--lineage-config",
            str(config_path),
            "--fingerprint",
            "proflingo",
            "llmmap",
            "--output",
            str(output_path),
        ],
    )

    assert verify_fingerprint_lineage.main() == 0
    report = json.loads(output_path.read_text())

    assert calls == [
        ("load", "org/base", "main"),
        ("proflingo", "base@main"),
        ("llmmap", "base@main"),
        ("cleanup", "org/base", "main"),
        ("load", "org/rl", "main"),
        ("proflingo", "rl@main"),
        ("llmmap", "rl@main"),
        ("cleanup", "org/rl", "main"),
        ("load", "org/rl", "step_200"),
        ("proflingo", "rl@step_200"),
        ("llmmap", "rl@step_200"),
        ("cleanup", "org/rl", "step_200"),
    ]
    assert report["lineage"] == "smoke_family"
    assert report["reference"]["model_id"] == "org/reference"
    assert report["targets"] == ["base@main", "rl@main", "rl@step_200"]
    assert report["target_metadata"]["rl@step_200"]["step"] == 200
    assert report["proflingo"]["rl@step_200"]["target"]["revision"] == "step_200"
    assert report["llmmap"]["rl@step_200"]["target"]["revision"] == "step_200"


def test_lineage_cli_uses_configured_artifact_paths(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    artifact_root = tmp_path / "fingerprints"
    config_path.write_text(
        f"""
name: custom_artifacts
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated_reference.txt
    trap_suffixes: trap/custom_suffixes.csv
    llmmap_templates: llmmap/custom_templates.json
targets:
  - label: base
    model_id: org/base
""".strip()
    )
    captured = {}

    def fake_load_cases(path, *_args, **_kwargs):
        captured.setdefault("case_paths", []).append(path)
        return []

    monkeypatch.setattr(verify_fingerprint_lineage, "load_proflingo_cases", fake_load_cases)
    monkeypatch.setattr(verify_fingerprint_lineage, "load_trap_cases", fake_load_cases)
    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_model", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_for_target", lambda **_kwargs: {})
    monkeypatch.setattr(verify_fingerprint_lineage, "run_trap_for_target", lambda **_kwargs: {})
    monkeypatch.setattr(verify_fingerprint_lineage, "cleanup_after_target_model", lambda *_args: None)
    monkeypatch.setattr(verify_fingerprint_lineage, "write_json", lambda _path, _data: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_fingerprint_lineage.py",
            "--lineage-config",
            str(config_path),
            "--fingerprint",
            "proflingo",
            "trap",
            "--output",
            str(tmp_path / "report.json"),
        ],
    )

    assert verify_fingerprint_lineage.main() == 0
    assert captured == {
        "case_paths": [
            artifact_root / "proflingo/generated_reference.txt",
            artifact_root / "trap/custom_suffixes.csv",
        ]
    }


def test_lineage_cli_requires_artifact_paths_for_requested_fingerprints(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    config_path.write_text(
        """
name: missing_artifacts
reference:
  model_id: org/reference
  artifact_root: artifacts/fingerprints/reference
targets:
  - label: base
    model_id: org/base
""".strip()
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_fingerprint_lineage.py",
            "--lineage-config",
            str(config_path),
            "--fingerprint",
            "proflingo",
        ],
    )

    try:
        verify_fingerprint_lineage.main()
    except ValueError as exc:
        assert "proflingo_fingerprint" in str(exc)
    else:
        raise AssertionError("missing requested artifact path should be rejected")


def test_technique_selection_runs_only_requested_fingerprints(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    artifact_root = tmp_path / "fingerprints"
    for technique in ("proflingo", "trap", "llmmap"):
        (artifact_root / technique).mkdir(parents=True, exist_ok=True)
    (artifact_root / "proflingo/generated.txt").write_text("0,suffix\n")
    (artifact_root / "trap/suffixes.csv").write_text("goals,targets,control\n")
    (artifact_root / "llmmap/templates.json").write_text("{}")
    config_path.write_text(
        f"""
name: selection
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated.txt
    trap_suffixes: trap/suffixes.csv
    llmmap_templates: llmmap/templates.json
targets:
  - label: base
    model_id: org/base
""".strip()
    )

    def run_with_fingerprints(fingerprints: list[str]) -> list[str]:
        calls = []
        monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_model", lambda *_args, **_kwargs: (object(), object()))
        monkeypatch.setattr(verify_fingerprint_lineage, "load_proflingo_cases", lambda *_args, **_kwargs: ["p"])
        monkeypatch.setattr(verify_fingerprint_lineage, "load_trap_cases", lambda *_args, **_kwargs: ["t"])
        monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_for_target", lambda **_kwargs: calls.append("proflingo") or {})
        monkeypatch.setattr(verify_fingerprint_lineage, "run_trap_for_target", lambda **_kwargs: calls.append("trap") or {})
        monkeypatch.setattr(verify_fingerprint_lineage, "run_llmmap_for_target", lambda **_kwargs: calls.append("llmmap") or {})
        monkeypatch.setattr(verify_fingerprint_lineage, "cleanup_after_target_model", lambda *_args: None)
        monkeypatch.setattr(verify_fingerprint_lineage, "write_json", lambda _path, _data: None)
        monkeypatch.setattr(
            "sys.argv",
            [
                "verify_fingerprint_lineage.py",
                "--lineage-config",
                str(config_path),
                "--fingerprint",
                *fingerprints,
                "--output",
                str(tmp_path / f"{'-'.join(fingerprints)}.json"),
            ],
        )

        assert verify_fingerprint_lineage.main() == 0
        return calls

    assert run_with_fingerprints(["proflingo"]) == ["proflingo"]
    assert run_with_fingerprints(["trap"]) == ["trap"]
    assert run_with_fingerprints(["llmmap"]) == ["llmmap"]
    assert run_with_fingerprints(["proflingo", "trap", "llmmap"]) == [
        "proflingo",
        "trap",
        "llmmap",
    ]


def test_cleanup_runs_when_technique_raises(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    artifact_root = tmp_path / "fingerprints"
    (artifact_root / "proflingo").mkdir(parents=True)
    (artifact_root / "proflingo/generated.txt").write_text("0,suffix\n")
    config_path.write_text(
        f"""
name: failure_cleanup
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated.txt
targets:
  - label: base
    model_id: org/base
""".strip()
    )
    calls = []

    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_model", lambda *_args, **_kwargs: ("model", "tokenizer"))
    monkeypatch.setattr(verify_fingerprint_lineage, "load_proflingo_cases", lambda *_args, **_kwargs: ["case"])

    def raise_from_proflingo(**_kwargs):
        calls.append("proflingo")
        raise RuntimeError("boom")

    monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_for_target", raise_from_proflingo)
    monkeypatch.setattr(
        verify_fingerprint_lineage,
        "cleanup_after_target_model",
        lambda model_id, revision: calls.append(("cleanup", model_id, revision)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_fingerprint_lineage.py",
            "--lineage-config",
            str(config_path),
            "--fingerprint",
            "proflingo",
        ],
    )

    try:
        verify_fingerprint_lineage.main()
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("technique failure should propagate")
    assert calls == ["proflingo", ("cleanup", "org/base", "main")]


def test_cleanup_after_target_model_evicts_entire_repo_cache(monkeypatch):
    calls = []

    monkeypatch.setattr(
        verify_fingerprint_lineage,
        "cleanup_torch_memory",
        lambda: calls.append("cleanup_torch_memory"),
    )
    monkeypatch.setattr(
        verify_fingerprint_lineage,
        "evict_hf_repo_cache",
        lambda model_id: calls.append(("evict_repo", model_id)),
    )

    verify_fingerprint_lineage.cleanup_after_target_model("org/model", "step_200")

    assert calls == ["cleanup_torch_memory", ("evict_repo", "org/model")]


def test_run_proflingo_for_target_passes_revision_metadata_and_model():
    target = ModelTarget(label="rl", model_id="org/rl", revision="checkpoint-200", step=200)
    result = verify_fingerprint_lineage.run_proflingo_for_target(
        target=target,
        cases=[],
        model=object(),
        tokenizer=object(),
        max_new_tokens=8,
        proflingo_match="prefix",
    )

    assert result["model"] == "rl@checkpoint-200"
    assert result["target"]["model_id"] == "org/rl"
    assert result["target"]["step"] == 200


def test_llmmap_for_target_uses_stable_key_revision_and_metadata(monkeypatch):
    config = LineageConfig(
        name="family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[],
    )
    target = ModelTarget(label="rl", model_id="org/rl", revision="checkpoint-200", step=200)
    args = type(
        "Args",
        (),
        {
            "llmmap_model_path": Path("llmmap/model"),
            "llmmap_templates": Path("llmmap/templates.json"),
            "llmmap_prompt_conf_path": Path("llmmap/prompts"),
            "llmmap_num_prompt_confs": 2,
            "llmmap_top_k": 5,
            "max_new_tokens": 8,
            "seed": 41,
            "dtype": "bf16",
            "device_map": "cpu",
        },
    )()
    captured = {}

    def fake_run_llmmap(*_positional, **kwargs):
        captured.update(kwargs)
        return {"target": kwargs["target_metadata"]}

    monkeypatch.setattr(verify_fingerprint_lineage, "run_llmmap_verification_for_loaded_model", fake_run_llmmap)

    result = verify_fingerprint_lineage.run_llmmap_for_target(
        args=args,
        config=config,
        target=target,
        model=object(),
        tokenizer=object(),
    )

    assert captured["result_key"] == "rl@checkpoint-200"
    assert captured["target_metadata"]["model_id"] == "org/rl"
    assert result["target"]["step"] == 200
