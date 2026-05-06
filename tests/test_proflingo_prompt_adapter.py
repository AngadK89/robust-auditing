import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROFLINGO_DIR = ROOT_DIR / "third_party" / "ProFLingo"
sys.path.insert(0, str(PROFLINGO_DIR))

import proflingo


class PlainTokenizer:
    chat_template = None

    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("plain tokenizers should not use chat templates")


class ChatTokenizer:
    chat_template = "{{ messages }}"


def test_proflingo_uses_tokenizer_chat_template_capability():
    assert proflingo.uses_tokenizer_chat_template("allenai/OLMo-2-0425-1B", PlainTokenizer()) is False
    assert proflingo.uses_tokenizer_chat_template("plain-model", ChatTokenizer()) is True
