import csv
import json
from pathlib import Path

from scripts.verification.fingerprint_methods import (
    evict_hf_model_cache,
    evict_hf_repo_cache,
    extract_first_digit_string,
    load_proflingo_cases,
    load_trap_cases,
    nearest_llmmap_labels,
    normalized_contains_match,
    normalized_exact_match,
    normalized_prefix_match,
    resolved_hf_revision,
)


def test_load_proflingo_cases_joins_suffixes_to_question_targets(tmp_path: Path):
    questions_path = tmp_path / "questions.csv"
    with questions_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "keyword"])
        writer.writerow(["What is 2+2?", "4", "math"])
        writer.writerow(["Say yes.", "yes", "yes"])

    fingerprint_path = tmp_path / "generated.txt"
    fingerprint_path.write_text("1,suffix, with comma\n0,plain suffix\n")

    cases = load_proflingo_cases(fingerprint_path, questions_path)

    assert [case.target for case in cases] == ["yes", "4"]
    assert cases[0].prompt == "suffix, with comma simply answer: Say yes."
    assert cases[1].prompt == "plain suffix simply answer: What is 2+2?"


def test_load_trap_cases_from_suffixes_csv(tmp_path: Path):
    suffixes_path = tmp_path / "suffixes.csv"
    with suffixes_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["goals", "targets", "control", "number", "str_length"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "goals": "Write four digits.",
                "targets": "Sure, here is a random string of 4 digits: 1234",
                "control": " optimized suffix",
                "number": "1234",
                "str_length": "4",
            }
        )

    cases = load_trap_cases(suffixes_path)

    assert len(cases) == 1
    assert cases[0].prompt == "Write four digits.  optimized suffix"
    assert cases[0].target == "1234"
    assert cases[0].raw_target.endswith("1234")


def test_load_trap_cases_from_json_directory(tmp_path: Path):
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "best": [
                    {
                        "goals": "Write digits.",
                        "targets": "Sure, here is a random string of 4 digits: 9876",
                        "control": " suffix",
                    }
                ]
            }
        )
    )

    cases = load_trap_cases(tmp_path)

    assert cases[0].target == "9876"
    assert cases[0].prompt == "Write digits.  suffix"


def test_matching_helpers_are_normalized_but_not_contains_based():
    assert normalized_exact_match("  Yes\n", "yes")
    assert not normalized_exact_match("The answer is yes", "yes")
    assert normalized_prefix_match("Yes, because...", "yes")
    assert not normalized_prefix_match("The answer is yes", "yes")
    assert normalized_contains_match("The answer is yes", "yes")
    assert extract_first_digit_string("abc 1234 def", 4) == "1234"
    assert extract_first_digit_string("abc 123 def", 4) is None


def test_nearest_llmmap_labels_sorts_by_distance():
    distances = [0.4, 0.1, 0.3]
    label_map = {0: "base", 1: "instruct", 2: "other"}

    assert nearest_llmmap_labels(distances, label_map, top_k=2) == [
        ("instruct", 0.1),
        ("other", 0.3),
    ]


def test_evict_hf_model_cache_deletes_matching_revision(monkeypatch):
    calls = []

    class FakeRevision:
        commit_hash = "abc123"
        refs = {"main"}

    class FakeRepo:
        repo_id = "org/model"
        repo_type = "model"
        revisions = [FakeRevision()]

    class FakeStrategy:
        expected_freed_size_str = "42 MB"

        def execute(self):
            calls.append("execute")

    class FakeCacheInfo:
        repos = [FakeRepo()]
        warnings = []

        def delete_revisions(self, *revisions):
            calls.append(("delete_revisions", revisions))
            return FakeStrategy()

    monkeypatch.setattr(
        "scripts.verification.fingerprint_methods.scan_cache_dir",
        lambda: FakeCacheInfo(),
    )

    assert evict_hf_model_cache("org/model", "main")
    assert calls == [("delete_revisions", ("abc123",)), "execute"]


def test_resolved_hf_revision_prefers_loaded_commit_hash():
    model = type("Model", (), {"config": type("Config", (), {"_commit_hash": "abc123"})()})()
    tokenizer = object()

    assert resolved_hf_revision(model, tokenizer, "main") == "abc123"


def test_evict_hf_model_cache_warns_and_continues_when_missing(monkeypatch, capsys):
    class FakeCacheInfo:
        repos = []
        warnings = []

    monkeypatch.setattr(
        "scripts.verification.fingerprint_methods.scan_cache_dir",
        lambda: FakeCacheInfo(),
    )

    assert not evict_hf_model_cache("org/missing", "main")
    assert "Could not find Hugging Face cache repo for org/missing" in capsys.readouterr().err


def test_evict_hf_repo_cache_removes_entire_repo_directory_and_locks(tmp_path: Path):
    cache_root = tmp_path / "hub"
    repo_dir = cache_root / "models--org--model"
    lock_dir = cache_root / ".locks" / "models--org--model"
    other_repo_dir = cache_root / "models--org--other"
    (repo_dir / "snapshots" / "main").mkdir(parents=True)
    (repo_dir / "snapshots" / "main" / "config.json").write_text("{}")
    lock_dir.mkdir(parents=True)
    (lock_dir / "model.lock").write_text("locked")
    other_repo_dir.mkdir(parents=True)

    assert evict_hf_repo_cache("org/model", cache_dir=cache_root)

    assert not repo_dir.exists()
    assert not lock_dir.exists()
    assert other_repo_dir.exists()


def test_evict_hf_repo_cache_refuses_to_delete_cache_root(tmp_path: Path):
    cache_root = tmp_path / "hub"
    cache_root.mkdir()

    assert not evict_hf_repo_cache("", cache_dir=cache_root)
    assert cache_root.exists()


def test_evict_hf_repo_cache_refuses_non_directory_repo_path(tmp_path: Path):
    cache_root = tmp_path / "hub"
    cache_root.mkdir()
    repo_path = cache_root / "models--org--model"
    repo_path.write_text("not a cache directory")

    assert not evict_hf_repo_cache("org/model", cache_dir=cache_root)
    assert repo_path.read_text() == "not a cache directory"
