import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROFLINGO_DIR = ROOT_DIR / "third_party" / "ProFLingo"
sys.path.insert(0, str(PROFLINGO_DIR))

import pytest
import torch

from attack import _roundtrip_ids, _sample_roundtrip_prompt_ids, assemble_ids


class PrefixTokenizer:
    def __init__(self, prefix_ids):
        self.prefix_ids = prefix_ids

    def encode(self, text, add_special_tokens=True):
        ids = [ord(char) for char in text]
        if add_special_tokens:
            return self.prefix_ids + ids
        return ids


def test_roundtrip_ids_uses_tokenizer_no_special_tokens_when_available():
    tokenizer = PrefixTokenizer(prefix_ids=[101])

    assert _roundtrip_ids(tokenizer, "abc") == [97, 98, 99]


def test_roundtrip_ids_does_not_assume_two_special_tokens():
    tokenizer = PrefixTokenizer(prefix_ids=[])

    assert _roundtrip_ids(tokenizer, "abc") == [97, 98, 99]


class NeverRoundtripTokenizer(PrefixTokenizer):
    def encode(self, text, add_special_tokens=True):
        return [0]

    def decode(self, _ids):
        return "abc"


class AlwaysRoundtripTokenizer(PrefixTokenizer):
    def decode(self, ids):
        return "".join(chr(token_id) for token_id in ids)


def test_sample_roundtrip_prompt_ids_allows_empty_filter_word():
    repl_ids = torch.tensor([97, 98, 99])

    prompt_ids = _sample_roundtrip_prompt_ids(
        AlwaysRoundtripTokenizer(prefix_ids=[]),
        repl_ids,
        token_nums=2,
        filter_word="",
        device="cpu",
        max_attempts=3,
    )

    assert len(prompt_ids) == 2


def test_sample_roundtrip_prompt_ids_fails_instead_of_looping_forever():
    repl_ids = torch.tensor([97, 98, 99])

    with pytest.raises(RuntimeError, match="Unable to sample"):
        _sample_roundtrip_prompt_ids(
            NeverRoundtripTokenizer(prefix_ids=[]),
            repl_ids,
            token_nums=2,
            filter_word="",
            device="cpu",
            max_attempts=3,
        )


class PromptTemplate:
    roles = ("USER", "ASSISTANT")
    sep_style = object()

    def __init__(self):
        self.messages = []

    def append_message(self, role, message):
        self.messages.append([role, message])

    def update_last_message(self, message):
        self.messages[-1][1] = message

    def get_prompt(self):
        return "".join(f"{role}: {message or ''}\n" for role, message in self.messages)


class NoUnkIdTokenizer(PrefixTokenizer):
    unk_token_id = 100257


def test_assemble_ids_does_not_require_placeholder_to_encode_as_unk_id():
    begin_ids, _middle_ids, _target_ids = assemble_ids(
        NoUnkIdTokenizer(prefix_ids=[]),
        PromptTemplate(),
        question="Question",
        target="Answer",
    )

    assert begin_ids == [ord(char) for char in "USER: "]


class ContextSensitivePlaceholderTokenizer(PrefixTokenizer):
    def encode(self, text, add_special_tokens=True):
        ids = []
        index = 0
        while index < len(text):
            if text.startswith(" <", index):
                ids.append(1000)
                index += 2
            else:
                ids.append(ord(text[index]))
                index += 1
        if add_special_tokens:
            return self.prefix_ids + ids
        return ids


def test_assemble_ids_handles_context_sensitive_placeholder_tokens():
    begin_ids, _middle_ids, _target_ids = assemble_ids(
        ContextSensitivePlaceholderTokenizer(prefix_ids=[]),
        PromptTemplate(),
        question="Question",
        target="Answer",
    )

    assert begin_ids == [ord(char) for char in "USER: "]
