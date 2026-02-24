# Class AIMessage (defined in ragu/common/prompts/messages.py at lines 89-95)

@dataclasses.dataclass(frozen=True, slots=True)
class AIMessage(ragu.common.prompts.messages.BaseMessage):
    """
    Assistant (LLM) response message.
    """
    ...

    role: ragu.common.prompts.messages.Role = "assistant"