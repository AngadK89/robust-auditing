import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROFLINGO_DIR = ROOT_DIR / "third_party" / "ProFLingo"
sys.path.insert(0, str(PROFLINGO_DIR))

import proflingo
import copyright_test


class PlainTokenizer:
    chat_template = None

    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("plain tokenizers should not use chat templates")


class ChatTokenizer:
    chat_template = "{{ messages }}"


def test_proflingo_uses_tokenizer_chat_template_capability():
    assert proflingo.uses_tokenizer_chat_template("allenai/OLMo-2-0425-1B", PlainTokenizer()) is False
    assert proflingo.uses_tokenizer_chat_template("plain-model", ChatTokenizer()) is True


def test_default_templates_use_fallbacks_for_plain_tokenizers():
    tokenizer = PlainTokenizer()

    templates = proflingo.get_default_templates(tokenizer)

    assert [template.name for template in templates] == ["alpaca", "zero_shot"]
    assert templates[0].sep == " "
    assert templates[1].sep == "\n"


def test_default_templates_use_base_fingerprint_templates_for_chat_tokenizers():
    templates = proflingo.get_default_templates(ChatTokenizer())

    assert [template.name for template in templates] == ["alpaca", "zero_shot"]
    assert templates[0].sep == " "
    assert templates[1].sep == "\n"


def test_copyright_fingerprint_test_accepts_template_list_and_limit(monkeypatch, tmp_path):
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text(
        "question,answer,keyword\n"
        "Where does the sun rise?,east,east\n"
        "What color is grass?,green,green\n",
        encoding="utf-8",
    )
    advsamples_path = tmp_path / "generated.txt"
    advsamples_path.write_text("0,prefix A\n1,prefix B\n", encoding="utf-8")
    calls = []

    def fake_complete_conversation(model, tokenizer, template, question, size):
        calls.append((template, question, size))
        return "wrong" if template == "bad-template" else "The answer is east."

    monkeypatch.setattr(copyright_test, "complete_conversation", fake_complete_conversation)

    total, matched = copyright_test.fingerprint_test(
        model=object(),
        tokenizer=object(),
        dataset_path=questions_path,
        advsamples_path=advsamples_path,
        manual_check=False,
        model_path="org/plain-model",
        template=["bad-template", "good-template"],
        verbose=False,
        max_token=9,
        limit=1,
    )

    assert (total, matched) == (1, 1)
    assert calls == [
        ("bad-template", "prefix A simply answer: Where does the sun rise?", 9),
        ("good-template", "prefix A simply answer: Where does the sun rise?", 9),
    ]


def test_copyright_fingerprint_test_can_force_local_backend_for_gpt_named_hf_models(
    monkeypatch, tmp_path
):
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text(
        "question,answer,keyword\nWhere does the sun rise?,east,east\n",
        encoding="utf-8",
    )
    advsamples_path = tmp_path / "generated.txt"
    advsamples_path.write_text("0,prefix\n", encoding="utf-8")
    calls = []

    def fake_complete_conversation(model, tokenizer, template, question, size):
        calls.append((model, tokenizer, template, question, size))
        return "east"

    def forbidden_openai(*_args, **_kwargs):
        raise AssertionError("local HF model ids containing gpt must not call OpenAI")

    monkeypatch.setattr(copyright_test, "complete_conversation", fake_complete_conversation)
    monkeypatch.setattr(copyright_test, "get_answer_openai", forbidden_openai)

    total, matched = copyright_test.fingerprint_test(
        model="hf-model",
        tokenizer="hf-tokenizer",
        dataset_path=questions_path,
        advsamples_path=advsamples_path,
        manual_check=False,
        model_path="openai-community/gpt2",
        template=["template"],
        verbose=False,
        backend="local",
    )

    assert (total, matched) == (1, 1)
    assert calls == [
        (
            "hf-model",
            "hf-tokenizer",
            "template",
            "prefix simply answer: Where does the sun rise?",
            64,
        )
    ]
