from __future__ import annotations

from functools import lru_cache


OLMO2_TEMPLATE_NAME = "olmo2"
OLMO2_CHAT_MODEL_MARKERS = (
    "OLMo-2-0425-1B-SFT",
    "OLMo-2-0425-1B-DPO",
    "OLMo-2-0425-1B-RLVR1",
    "OLMo-2-0425-1B-Instruct",
)


def _is_olmo2_chat_model(model_path: str) -> bool:
    normalized = model_path.replace("\\", "/")
    return any(marker in normalized for marker in OLMO2_CHAT_MODEL_MARKERS)


@lru_cache(maxsize=1)
def register_olmo2_fastchat_support() -> None:
    from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template
    from fastchat.model import model_adapter
    from fastchat.model.model_adapter import BaseModelAdapter

    register_conv_template(
        Conversation(
            name=OLMO2_TEMPLATE_NAME,
            system_template="<|endoftext|>{system_message}",
            system_message="",
            roles=("<|user|>\n", "<|assistant|>\n"),
            sep_style=SeparatorStyle.NO_COLON_TWO,
            sep="\n",
            sep2="<|endoftext|>\n",
            stop_token_ids=[100257],
        ),
        override=True,
    )

    class OLMo2ModelAdapter(BaseModelAdapter):
        def match(self, model_path: str):
            return _is_olmo2_chat_model(model_path)

        def get_default_conv_template(self, model_path: str):
            from fastchat.conversation import get_conv_template

            return get_conv_template(OLMO2_TEMPLATE_NAME)

    if not any(type(adapter).__name__ == "OLMo2ModelAdapter" for adapter in model_adapter.model_adapters):
        model_adapter.register_model_adapter(OLMo2ModelAdapter)
        model_adapter.get_model_adapter.cache_clear()


def build_olmo2_prompt_for_test(user_turns: list[str]) -> str:
    from fastchat.model import get_conversation_template

    register_olmo2_fastchat_support()
    conv = get_conversation_template("allenai/OLMo-2-0425-1B-Instruct")
    for index, user_turn in enumerate(user_turns):
        conv.append_message(conv.roles[0], user_turn)
        conv.append_message(conv.roles[1], None)
        if index < len(user_turns) - 1:
            conv.update_last_message(f"{['First', 'Second', 'Third'][index]} answer")
    return conv.get_prompt()

