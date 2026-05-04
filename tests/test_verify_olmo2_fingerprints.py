import csv
import json
from pathlib import Path

from scripts.verification.verify_olmo2_fingerprints import (
    ALL_FINGERPRINTS,
    build_parser,
    extract_first_digit_string,
    load_proflingo_cases,
    load_trap_cases,
    main,
    nearest_llmmap_labels,
    normalized_contains_match,
    normalized_exact_match,
    normalized_prefix_match,
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


def test_build_parser_requires_model():
    parser = build_parser()

    try:
        parser.parse_args([])
    except SystemExit:
        return
    raise AssertionError("--model should be required")


def test_build_parser_defaults_to_all_fingerprints():
    args = build_parser().parse_args(["--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"])

    assert args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert args.fingerprint == list(ALL_FINGERPRINTS)


def test_build_parser_accepts_single_fingerprint_selection():
    args = build_parser().parse_args(
        ["--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--fingerprint", "proflingo"]
    )

    assert args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert args.fingerprint == ["proflingo"]


def test_build_parser_accepts_multiple_fingerprint_selection():
    args = build_parser().parse_args(
        [
            "--model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--fingerprint",
            "proflingo",
            "llmmap",
        ]
    )

    assert args.fingerprint == ["proflingo", "llmmap"]


def test_build_parser_rejects_removed_skip_options():
    parser = build_parser()

    for removed_flag in ("--skip-adversarial", "--skip-llmmap"):
        try:
            parser.parse_args([removed_flag])
        except SystemExit:
            continue
        raise AssertionError(f"{removed_flag} should not be accepted")


def test_main_runs_only_selected_fingerprints(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    output = tmp_path / "report.json"

    def fake_run_replay(args):
        calls.append("replay")
        assert args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert args.fingerprint == ["proflingo"]
        return {"proflingo": {args.model: {"match_rate": 1.0}}}

    def fake_run_llmmap(*_args, **_kwargs):
        calls.append("llmmap")
        return {}

    written = {}

    def fake_write_json(path, data):
        written["path"] = path
        written["data"] = data

    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_olmo2_fingerprints.py",
            "--model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--fingerprint",
            "proflingo",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        "scripts.verification.verify_olmo2_fingerprints.run_replay_verification",
        fake_run_replay,
    )
    monkeypatch.setattr(
        "scripts.verification.verify_olmo2_fingerprints.run_llmmap_verification",
        fake_run_llmmap,
    )
    monkeypatch.setattr(
        "scripts.verification.verify_olmo2_fingerprints.write_json",
        fake_write_json,
    )

    assert main() == 0
    assert calls == ["replay"]
    assert written["path"] == output
    assert written["data"]["model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert written["data"]["fingerprint"] == ["proflingo"]
