import inspect
import json
import sys
import types
from pathlib import Path

from scripts.verification.fingerprint_methods import (
    evict_hf_model_cache,
    evict_hf_repo_cache,
    load_hf_tokenizer,
    nearest_llmmap_labels,
    resolved_hf_revision,
    run_llmmap_verification_for_loaded_model,
)


def test_load_hf_tokenizer_uses_left_padding_for_decoder_generation(monkeypatch):
    class FakeTokenizer:
        padding_side = "right"
        pad_token_id = None
        eos_token = "<eos>"

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return FakeTokenizer()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    tokenizer = load_hf_tokenizer("org/model")

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"


def test_nearest_llmmap_labels_sorts_by_distance():
    distances = [0.4, 0.1, 0.3]
    label_map = {0: "base", 1: "instruct", 2: "other"}

    assert nearest_llmmap_labels(distances, label_map, top_k=2) == [
        ("instruct", 0.1),
        ("other", 0.3),
    ]


def test_llmmap_verification_uses_direct_queries_and_loaded_templates(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}
    queries = ["Which token comes first?", "Name the marker."]

    class FakeLLMmap:
        def __init__(self):
            self.queries = queries
            self.conf = {}
            self.templates_map = {}
            self.label_map = {}
            self.DB = None

        def __call__(self, answers):
            calls["llmmap_answers"] = list(answers)
            return [
                0.05 if self.label_map[index] == "reference" else 2.0
                for index in range(len(self.label_map))
            ]

        def compute_template(self, entries):
            raise AssertionError("direct LLMmap verification must not compute target templates")

    fake_llmmap = FakeLLMmap()

    def fake_load_llmmap(path, device, verbose):
        calls["load_llmmap"] = {"path": path, "device": device, "verbose": verbose}
        return object(), fake_llmmap

    llmmap_pkg = types.ModuleType("LLMmap")
    inference_module = types.ModuleType("LLMmap.inference")
    inference_module.load_LLMmap = fake_load_llmmap
    dataset_module = types.ModuleType("LLMmap.dataset_maker")

    def forbidden_dataset_entries(*args, **kwargs):
        raise AssertionError("direct LLMmap verification must not sample prompt configurations")

    dataset_module.make_dataset_entries_for_new_llm = forbidden_dataset_entries
    prompt_module = types.ModuleType("LLMmap.prompt_configuration")

    class ForbiddenPromptConfFactory:
        def __init__(self, *args, **kwargs):
            raise AssertionError("direct LLMmap verification must not load PromptConfFactory")

    prompt_module.PromptConfFactory = ForbiddenPromptConfFactory
    prompt_module.TRAIN = "train"
    monkeypatch.setitem(sys.modules, "LLMmap", llmmap_pkg)
    monkeypatch.setitem(sys.modules, "LLMmap.inference", inference_module)
    monkeypatch.setitem(sys.modules, "LLMmap.dataset_maker", dataset_module)
    monkeypatch.setitem(sys.modules, "LLMmap.prompt_configuration", prompt_module)

    class FakeLocalHFLLM:
        def __init__(self, llm_name, model, tokenizer, max_new_tokens):
            calls["local_llm_init"] = {
                "llm_name": llm_name,
                "model": model,
                "tokenizer": tokenizer,
                "max_new_tokens": max_new_tokens,
            }

        def make_prompt(self, system, user):
            calls.setdefault("make_prompt_calls", []).append((system, user))
            assert system is None
            return f"PROMPT::{user}"

        def generate(self, prompt, gen_kargs):
            calls.setdefault("generate_calls", []).append((prompt, dict(gen_kargs)))
            return [f"ANSWER::{prompt.removeprefix('PROMPT::')}"]

    monkeypatch.setattr(
        "scripts.verification.fingerprint_methods.LocalHFLLM",
        FakeLocalHFLLM,
    )

    llmmap_model_path = tmp_path / "llmmap_model"
    llmmap_model_path.mkdir()
    llmmap_templates_path = tmp_path / "templates.json"
    llmmap_templates_path.write_text(
        json.dumps({"reference": [0.0, 0.0], "other": [1.0, 1.0]})
    )

    result = run_llmmap_verification_for_loaded_model(
        model_id="org/model",
        reference_model="reference",
        llmmap_model_path=llmmap_model_path,
        llmmap_templates_path=llmmap_templates_path,
        top_k=2,
        max_new_tokens=7,
        model=object(),
        tokenizer=object(),
        result_key="target-key",
        target_metadata={"revision": "abc123"},
    )

    assert calls["load_llmmap"] == {
        "path": str(llmmap_model_path),
        "device": "cpu",
        "verbose": False,
    }
    assert calls["local_llm_init"]["llm_name"] == "target-key"
    assert calls["local_llm_init"]["max_new_tokens"] == 7
    assert calls["make_prompt_calls"] == [(None, query) for query in queries]
    assert calls["generate_calls"] == [
        (f"PROMPT::{query}", {}) for query in queries
    ]
    answers = [f"ANSWER::{query}" for query in queries]
    assert calls["llmmap_answers"] == answers
    assert fake_llmmap.templates_map["reference"].tolist() == [0.0, 0.0]
    assert result["matched_reference_top1"] is True
    assert result["reference_model"] == "reference"
    assert result["verification_mode"] == "direct_queries"
    assert result["query_count"] == 2
    assert result["template_count"] == 2
    assert result["distance_fn"] == "euclidean"
    assert result["traces"] == [
        {"query": query, "response": answer} for query, answer in zip(queries, answers)
    ]
    assert result["top_k"] == [
        {"label": "reference", "distance": 0.05},
        {"label": "other", "distance": 2.0},
    ]
    assert result["target"] == {"revision": "abc123"}


def test_llmmap_loaded_model_signature_excludes_prompt_sampling_options():
    signature = inspect.signature(run_llmmap_verification_for_loaded_model)

    assert "prompt_conf_path" not in signature.parameters
    assert "num_prompt_confs" not in signature.parameters
    assert "seed" not in signature.parameters


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
