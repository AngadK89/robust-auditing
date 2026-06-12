import json
from pathlib import Path

import pytest

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


def test_load_lineage_config_parses_fingerprint_options(tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    config_path.write_text(
        """
name: custom_family
reference:
  model_id: org/reference
  artifact_root: artifacts/fingerprints/reference
fingerprints:
  proflingo:
    questions: custom/questions.csv
  llmmap:
    model_path: custom/llmmap/model
    top_k: 3
targets:
  - label: base
    model_id: org/base
""".strip()
    )

    config = load_lineage_config(config_path)

    assert config.fingerprints == {
        "proflingo": {
            "questions": Path("custom/questions.csv"),
        },
        "llmmap": {
            "model_path": Path("custom/llmmap/model"),
            "top_k": 3,
        },
    }


def test_load_lineage_config_defaults_fingerprint_options_to_empty_mapping(tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    config_path.write_text(
        """
name: custom_family
reference:
  model_id: org/reference
  artifact_root: artifacts/fingerprints/reference
targets:
  - label: base
    model_id: org/base
""".strip()
    )

    config = load_lineage_config(config_path)

    assert config.fingerprints == {}


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

    def fake_load_tokenizer(model_id, **kwargs):
        revision = kwargs.get("revision")
        calls.append(("load_slow_tokenizer", model_id, revision, kwargs.get("use_fast")))
        return f"slow-tokenizer:{model_id}:{revision}"

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
    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_tokenizer", fake_load_tokenizer)
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
        ("load_slow_tokenizer", "org/base", "main", False),
        ("proflingo", "base@main"),
        ("llmmap", "base@main"),
        ("cleanup", "org/base", "main"),
        ("load", "org/rl", "main"),
        ("load_slow_tokenizer", "org/rl", "main", False),
        ("proflingo", "rl@main"),
        ("llmmap", "rl@main"),
        ("cleanup", "org/rl", "main"),
        ("load", "org/rl", "step_200"),
        ("load_slow_tokenizer", "org/rl", "step_200", False),
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


def test_lineage_cli_uses_yaml_proflingo_options(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    artifact_root = tmp_path / "fingerprints"
    questions_path = tmp_path / "custom_questions.csv"
    (artifact_root / "proflingo").mkdir(parents=True)
    (artifact_root / "proflingo/generated.txt").write_text("0,suffix\n")
    questions_path.write_text("question,answer\nq,a\n")
    config_path.write_text(
        f"""
name: proflingo_options
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated.txt
fingerprints:
  proflingo:
    questions: {questions_path}
targets:
  - label: base
    model_id: org/base
""".strip()
    )
    captured = {}

    def fake_run_proflingo(**kwargs):
        captured["fingerprint_path"] = kwargs["fingerprint_path"]
        captured["questions_path"] = kwargs["questions_path"]
        captured["limit"] = kwargs["limit"]
        return {}

    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_model", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_for_target", fake_run_proflingo)
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
            "--limit",
            "7",
        ],
    )

    assert verify_fingerprint_lineage.main() == 0
    assert captured == {
        "fingerprint_path": artifact_root / "proflingo/generated.txt",
        "questions_path": questions_path,
        "limit": 7,
    }


def test_lineage_cli_defaults_to_configured_output(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    output_path = tmp_path / "configured_report.json"
    artifact_root = tmp_path / "fingerprints"
    (artifact_root / "proflingo").mkdir(parents=True)
    (artifact_root / "proflingo/generated.txt").write_text("0,suffix\n")
    config_path.write_text(
        f"""
name: configured_output
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated.txt
output: {output_path}
targets:
  - label: base
    model_id: org/base
""".strip()
    )

    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_model", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_for_target", lambda **_kwargs: {})
    monkeypatch.setattr(verify_fingerprint_lineage, "cleanup_after_target_model", lambda *_args: None)

    written = {}
    monkeypatch.setattr(verify_fingerprint_lineage, "write_json", lambda path, data: written.update(path=path, data=data))
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

    assert verify_fingerprint_lineage.main() == 0
    assert written["path"] == output_path


def test_lineage_cli_defaults_to_fingerprints_declared_by_artifacts(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    artifact_root = tmp_path / "fingerprints"
    (artifact_root / "proflingo").mkdir(parents=True)
    (artifact_root / "llmmap").mkdir()
    (artifact_root / "proflingo/generated.txt").write_text("0,suffix\n")
    (artifact_root / "llmmap/templates.json").write_text("{}")
    config_path.write_text(
        f"""
name: configured_fingerprints
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated.txt
    llmmap_templates: llmmap/templates.json
targets:
  - label: base
    model_id: org/base
""".strip()
    )

    calls = []
    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_model", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_for_target", lambda **_kwargs: calls.append("proflingo") or {})
    monkeypatch.setattr(verify_fingerprint_lineage, "run_llmmap_for_target", lambda **_kwargs: calls.append("llmmap") or {})
    monkeypatch.setattr(verify_fingerprint_lineage, "cleanup_after_target_model", lambda *_args: None)
    monkeypatch.setattr(verify_fingerprint_lineage, "write_json", lambda _path, _data: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_fingerprint_lineage.py",
            "--lineage-config",
            str(config_path),
        ],
    )

    assert verify_fingerprint_lineage.main() == 0
    assert calls == ["proflingo", "llmmap"]


def test_lineage_cli_uses_yaml_llmmap_options(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "lineage.yaml"
    artifact_root = tmp_path / "fingerprints"
    (artifact_root / "llmmap").mkdir(parents=True)
    (artifact_root / "llmmap/templates.json").write_text("{}")
    config_path.write_text(
        f"""
name: llmmap_options
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    llmmap_templates: llmmap/templates.json
fingerprints:
  llmmap:
    model_path: custom/llmmap/model
    top_k: 3
targets:
  - label: base
    model_id: org/base
""".strip()
    )
    captured = {}

    def fake_run_llmmap(*positional, **kwargs):
        captured["positional"] = positional
        captured.update(kwargs)
        return {"target": kwargs["target_metadata"]}

    loaded_model = object()
    loaded_tokenizer = object()

    monkeypatch.setattr(
        verify_fingerprint_lineage,
        "load_hf_model",
        lambda *_args, **_kwargs: (loaded_model, loaded_tokenizer),
    )
    monkeypatch.setattr(verify_fingerprint_lineage, "run_llmmap_verification_for_loaded_model", fake_run_llmmap)
    monkeypatch.setattr(verify_fingerprint_lineage, "cleanup_after_target_model", lambda *_args: None)
    monkeypatch.setattr(verify_fingerprint_lineage, "write_json", lambda _path, _data: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_fingerprint_lineage.py",
            "--lineage-config",
            str(config_path),
            "--fingerprint",
            "llmmap",
        ],
    )

    assert verify_fingerprint_lineage.main() == 0
    assert captured["positional"] == (
        "org/base",
        "org/reference",
        Path("custom/llmmap/model"),
        artifact_root / "llmmap/templates.json",
        3,
        64,
        loaded_model,
        loaded_tokenizer,
    )


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
    for technique in ("proflingo", "llmmap"):
        (artifact_root / technique).mkdir(parents=True, exist_ok=True)
    (artifact_root / "proflingo/generated.txt").write_text("0,suffix\n")
    (artifact_root / "llmmap/templates.json").write_text("{}")
    config_path.write_text(
        f"""
name: selection
reference:
  model_id: org/reference
  artifact_root: {artifact_root}
  artifacts:
    proflingo_fingerprint: proflingo/generated.txt
    llmmap_templates: llmmap/templates.json
targets:
  - label: base
    model_id: org/base
""".strip()
    )

    def run_with_fingerprints(fingerprints: list[str]) -> list[str]:
        calls = []
        monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_model", lambda *_args, **_kwargs: (object(), object()))
        monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_tokenizer", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_for_target", lambda **_kwargs: calls.append("proflingo") or {})
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
    assert run_with_fingerprints(["llmmap"]) == ["llmmap"]
    assert run_with_fingerprints(["proflingo", "llmmap"]) == [
        "proflingo",
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
    monkeypatch.setattr(verify_fingerprint_lineage, "load_hf_tokenizer", lambda *_args, **_kwargs: "slow-tokenizer")

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


def test_run_proflingo_for_target_passes_revision_metadata_and_model(monkeypatch, tmp_path):
    target = ModelTarget(label="rl", model_id="org/rl", revision="checkpoint-200", step=200)
    fingerprint_path = tmp_path / "generated.txt"
    questions_path = tmp_path / "questions.csv"
    fingerprint_path.write_text("0,suffix\n", encoding="utf-8")
    questions_path.write_text("question,answer,keyword\nq,a,a\n", encoding="utf-8")
    calls = {}

    def fake_default_templates(model_id):
        calls["template_model_id"] = model_id
        return "template-a"

    def fake_copyright_test(**kwargs):
        calls.update(kwargs)
        calls["advsamples_text"] = Path(kwargs["advsamples_path"]).read_text(encoding="utf-8")
        return 5, 3

    monkeypatch.setattr(verify_fingerprint_lineage, "get_proflingo_default_templates", fake_default_templates)
    monkeypatch.setattr(verify_fingerprint_lineage, "run_proflingo_copyright_test", fake_copyright_test)

    result = verify_fingerprint_lineage.run_proflingo_for_target(
        target=target,
        fingerprint_path=fingerprint_path,
        questions_path=questions_path,
        model="loaded-model",
        tokenizer="slow-tokenizer",
        max_new_tokens=8,
        limit=11,
    )

    assert result["model"] == "rl@checkpoint-200"
    assert result["technique"] == "proflingo"
    assert result["total"] == 5
    assert result["matched"] == 3
    assert result["match_rate"] == 0.6
    assert result["verification_mode"] == "proflingo_copyright_test"
    assert result["target"]["model_id"] == "org/rl"
    assert result["target"]["step"] == 200
    assert calls["model"] == "loaded-model"
    assert calls["tokenizer"] == "slow-tokenizer"
    assert calls["dataset_path"] == questions_path
    assert calls["advsamples_text"] == "0,suffix\n"
    assert calls["manual_check"] is False
    assert calls["model_path"] == "org/rl"
    assert calls["template"] == "template-a"
    assert calls["verbose"] is False
    assert calls["max_token"] == 8
    assert calls["template_model_id"] == "org/rl"


def test_get_proflingo_default_templates_uses_olmo_chat_template_sentinel():
    assert (
        verify_fingerprint_lineage.get_proflingo_default_templates(
            "allenai/OLMo-2-0425-1B-Instruct"
        )
        is None
    )


def test_llmmap_for_target_uses_yaml_options_stable_key_revision_and_metadata(monkeypatch):
    config = LineageConfig(
        name="family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[],
        fingerprints={
            "llmmap": {
                "model_path": Path("llmmap/model"),
                "top_k": 5,
            }
        },
    )
    target = ModelTarget(label="rl", model_id="org/rl", revision="checkpoint-200", step=200)
    args = type(
        "Args",
        (),
        {
            "llmmap_templates": Path("llmmap/templates.json"),
            "max_new_tokens": 8,
            "dtype": "bf16",
            "device_map": "cpu",
        },
    )()
    captured = {}

    def fake_run_llmmap(*_positional, **kwargs):
        captured["_positional"] = _positional
        captured.update(kwargs)
        return {"target": kwargs["target_metadata"]}

    monkeypatch.setattr(verify_fingerprint_lineage, "run_llmmap_verification_for_loaded_model", fake_run_llmmap)

    loaded_model = object()
    loaded_tokenizer = object()

    result = verify_fingerprint_lineage.run_llmmap_for_target(
        args=args,
        config=config,
        target=target,
        model=loaded_model,
        tokenizer=loaded_tokenizer,
    )

    assert captured["_positional"] == (
        "org/rl",
        "org/reference",
        Path("llmmap/model"),
        Path("llmmap/templates.json"),
        5,
        8,
        loaded_model,
        loaded_tokenizer,
    )
    assert captured["result_key"] == "rl@checkpoint-200"
    assert captured["target_metadata"]["model_id"] == "org/rl"
    assert result["target"]["step"] == 200


def test_llmmap_options_accepts_only_model_path_and_top_k():
    config = LineageConfig(
        name="family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[],
        fingerprints={
            "llmmap": {
                "model_path": "custom/llmmap/model",
                "top_k": 3,
            }
        },
    )

    options = verify_fingerprint_lineage.llmmap_options(config)

    assert options == verify_fingerprint_lineage.LLMmapOptions(
        model_path=Path("custom/llmmap/model"),
        top_k=3,
    )


def test_proflingo_options_rejects_stale_match_key():
    config = LineageConfig(
        name="family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[],
        fingerprints={"proflingo": {"questions": "questions.csv", "match": "exact"}},
    )

    with pytest.raises(ValueError, match="match"):
        verify_fingerprint_lineage.proflingo_options(config)


@pytest.mark.parametrize("stale_key", ["prompt_conf_path", "num_prompt_confs"])
def test_llmmap_options_rejects_stale_yaml_keys(stale_key):
    config = LineageConfig(
        name="family",
        reference=FingerprintReference(
            model_id="org/reference",
            artifact_root=Path("artifacts/fingerprints/reference"),
        ),
        targets=[],
        fingerprints={
            "llmmap": {
                "model_path": "custom/llmmap/model",
                "top_k": 3,
                stale_key: "stale",
            }
        },
    )

    with pytest.raises(ValueError, match=stale_key):
        verify_fingerprint_lineage.llmmap_options(config)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--proflingo-match", "exact"),
        ("--proflingo-questions", "questions.csv"),
        ("--llmmap-model-path", "llmmap/model"),
        ("--llmmap-prompt-conf-path", "llmmap/prompts"),
        ("--llmmap-num-prompt-confs", "2"),
        ("--llmmap-top-k", "5"),
        ("--seed", "41"),
    ],
)
def test_removed_cli_flags_are_rejected(flag, value):
    parser = verify_fingerprint_lineage.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--lineage-config", "lineage.yaml", flag, value])
