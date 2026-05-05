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


def test_lineage_cli_writes_report_with_target_metadata(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    output_path = tmp_path / "report.json"
    artifact_root = tmp_path / "artifacts"
    config_path.write_text(
        f"""
name: smoke_family
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated.txt
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

    def fake_run_replay(args, targets):
        calls.append(("replay", [target.key for target in targets]))
        return {
            "proflingo": {
                target.key: {"match_rate": 1.0, "target": target.to_report_dict()}
                for target in targets
            }
        }

    def fake_write_json(path, data):
        path.write_text(json.dumps(data, indent=2))

    monkeypatch.setattr(verify_fingerprint_lineage, "run_replay_verification_for_targets", fake_run_replay)
    monkeypatch.setattr(verify_fingerprint_lineage, "write_json", fake_write_json)
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_fingerprint_lineage.py",
            "--lineage-config",
            str(config_path),
            "--fingerprint",
            "proflingo",
            "--output",
            str(output_path),
        ],
    )

    assert verify_fingerprint_lineage.main() == 0
    report = json.loads(output_path.read_text())

    assert calls == [("replay", ["base@main", "rl@main", "rl@step_200"])]
    assert report["lineage"] == "smoke_family"
    assert report["reference"]["model_id"] == "org/reference"
    assert report["targets"] == ["base@main", "rl@main", "rl@step_200"]
    assert report["target_metadata"]["rl@step_200"]["step"] == 200
    assert report["proflingo"]["rl@step_200"]["target"]["revision"] == "step_200"


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

    def fake_run_replay(args, targets):
        captured["proflingo_fingerprint"] = args.proflingo_fingerprint
        captured["trap_suffixes"] = args.trap_suffixes
        return {"proflingo": {}}

    monkeypatch.setattr(verify_fingerprint_lineage, "run_replay_verification_for_targets", fake_run_replay)
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
        "proflingo_fingerprint": artifact_root / "proflingo/generated_reference.txt",
        "trap_suffixes": artifact_root / "trap/custom_suffixes.csv",
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


def test_replay_for_targets_passes_revision_and_metadata(monkeypatch):
    target = ModelTarget(label="rl", model_id="org/rl", revision="checkpoint-200", step=200)
    args = type(
        "Args",
        (),
        {
            "fingerprint": ["proflingo"],
            "proflingo_fingerprint": Path("proflingo/generated.txt"),
            "proflingo_questions": Path("questions.csv"),
            "trap_suffixes": None,
            "limit": None,
            "max_new_tokens": 8,
            "proflingo_match": "prefix",
            "dtype": "bf16",
            "device_map": "cpu",
        },
    )()
    captured = {}

    def fake_run_replay(specs, **kwargs):
        captured["specs"] = specs
        captured["kwargs"] = kwargs
        return {"proflingo": {}}

    monkeypatch.setattr(verify_fingerprint_lineage, "run_replay_for_model_specs", fake_run_replay)

    verify_fingerprint_lineage.run_replay_verification_for_targets(args, [target])

    spec = captured["specs"][0]
    assert spec.key == "rl@checkpoint-200"
    assert spec.model_id == "org/rl"
    assert spec.revision == "checkpoint-200"
    assert spec.metadata["step"] == 200
    assert captured["kwargs"]["dtype"] == "bf16"


def test_llmmap_for_targets_uses_stable_key_revision_and_metadata(monkeypatch):
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
        return {"rl@checkpoint-200": {"target": kwargs["target_metadata"]}}

    monkeypatch.setattr(verify_fingerprint_lineage, "run_llmmap_verification", fake_run_llmmap)

    result = verify_fingerprint_lineage.run_llmmap_verification_for_targets(
        args,
        config,
        [target],
    )

    assert captured["revision"] == "checkpoint-200"
    assert captured["result_key"] == "rl@checkpoint-200"
    assert captured["target_metadata"]["model_id"] == "org/rl"
    assert result["rl@checkpoint-200"]["target"]["step"] == 200
