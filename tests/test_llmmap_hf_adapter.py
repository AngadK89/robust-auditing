import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LLMMAP_DIR = ROOT_DIR / "third_party" / "LLMmap"
sys.path.insert(0, str(LLMMAP_DIR))

from LLMmap.llm import LLM_huggingface


class PlainTokenizer:
    chat_template = None

    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("plain tokenizers should not use chat templates")


class ChatTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "chat formatted prompt"


def test_hf_adapter_formats_plain_prompt_without_chat_template():
    llm = object.__new__(LLM_huggingface)
    llm.tokenizer = PlainTokenizer()

    prompt = llm.make_prompt("System guidance", "User query")

    assert prompt == "System guidance\n\nUser query"


def test_hf_adapter_keeps_chat_template_formatting_when_available():
    tokenizer = ChatTokenizer()
    llm = object.__new__(LLM_huggingface)
    llm.tokenizer = tokenizer

    prompt = llm.make_prompt("System guidance", "User query")

    assert prompt == "chat formatted prompt"
    assert tokenizer.messages == [{"role": "user", "content": "System guidance\n\nUser query"}]
    assert tokenizer.kwargs == {"tokenize": False, "add_generation_prompt": True}
