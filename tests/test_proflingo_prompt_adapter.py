import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROFLINGO_DIR = ROOT_DIR / "third_party" / "ProFLingo"
sys.path.insert(0, str(PROFLINGO_DIR))

import attack
import copyright_test


class TokenizerWithoutUnk:
    unk_token_id = None
    unk_token = None

    def encode(self, text, add_special_tokens=True):
        ids = [ord(character) for character in text]
        return [0] + ids if add_special_tokens else ids


class SimpleTemplate:
    roles = ("User", "Assistant")
    sep_style = None

    def __init__(self):
        self.messages = []

    def append_message(self, role, content):
        self.messages.append([role, content])

    def update_last_message(self, content):
        self.messages[-1][1] = content

    def get_prompt(self):
        return "".join(content or "" for _role, content in self.messages)


def decode(ids):
    return "".join(chr(token_id) for token_id in ids)


def test_assemble_ids_supports_tokenizers_without_unk_token():
    begin_ids, middle_ids, target_ids = attack.assemble_ids(
        TokenizerWithoutUnk(),
        SimpleTemplate(),
        question="Question?",
        target="Answer",
    )

    assert begin_ids == [0]
    assert decode(middle_ids) == "Question?"
    assert decode(target_ids) == "Answer"


def test_olmo2_base_uses_zero_shot_template():
    template = copyright_test.get_template("allenai/OLMo-2-0425-1B")

    assert template.name == "zero_shot"
    assert template.sep == "\n"


def test_olmo2_chat_models_use_tokenizer_default_chat_template():
    chat_model_ids = [
        "allenai/OLMo-2-0425-1B-SFT",
        "allenai/OLMo-2-0425-1B-DPO",
        "allenai/OLMo-2-0425-1B-RLVR1",
        "allenai/OLMo-2-0425-1B-Instruct",
        "allenai/OLMo-7B-Instruct",
    ]

    for model_id in chat_model_ids:
        assert copyright_test.get_template(model_id) is None
