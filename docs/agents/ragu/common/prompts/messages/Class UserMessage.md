# Class UserMessage (defined in ragu/common/prompts/messages.py at lines 81-87)

@dataclasses.dataclass(frozen=True, slots=True)
class UserMessage(ragu.common.prompts.messages.BaseMessage):
    """
    User input message.
    """
    ...

    role: ragu.common.prompts.messages.Role = "user"