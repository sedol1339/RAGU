# Class SystemMessage (defined in ragu/common/prompts/messages.py at lines 73-79)

@dataclasses.dataclass(frozen=True, slots=True)
class SystemMessage(ragu.common.prompts.messages.BaseMessage):
    """
    System-level instruction message.
    """
    ...

    role: ragu.common.prompts.messages.Role = "system"