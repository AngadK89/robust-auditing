import importlib.util
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
STRING_UTILS_PATH = (
    ROOT_DIR
    / "third_party"
    / "trap"
    / "llm_attacks"
    / "llm_attacks"
    / "minimal_gcg"
    / "string_utils.py"
)
spec = importlib.util.spec_from_file_location("trap_minimal_string_utils", STRING_UTILS_PATH)
string_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(string_utils)


class PlainEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class PlainTokenizer:
    chat_template = None

    def __call__(self, text):
        return PlainEncoding(list(range(len(text))))

    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("plain tokenizers should not use chat templates")


def test_olmo2_suffix_manager_supports_plain_tokenizer_without_chat_template():
    conv_template = string_utils.load_conversation_template("olmo2")
    manager = string_utils.SuffixManager(
        tokenizer=PlainTokenizer(),
        conv_template=conv_template,
        instruction="Goal",
        target="Target",
        adv_string="Control",
    )

    prompt = manager.get_prompt()

    assert prompt == "Goal ControlTarget"
    assert manager._goal_slice.start < manager._goal_slice.stop
    assert manager._control_slice.start < manager._control_slice.stop
    assert manager._target_slice.start < manager._target_slice.stop
