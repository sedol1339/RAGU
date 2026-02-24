# Class BaseMessage (defined in ragu/common/prompts/messages.py at lines 25-71)

@dataclasses.dataclass(frozen=True, slots=True)
class BaseMessage:
    """
    Base chat message abstraction.

    Represents a single message in a chat conversation with a fixed role
    (system, user, or assistant) and textual content. Provides conversion
    to OpenAI SDK message types.
    """
    ...

    content: str

    role: ragu.common.prompts.messages.Role

    name: str | None = None

    def to_openai(self) -> openai.types.chat.ChatCompletionMessageParam:
        """
        Convert this message into a typed OpenAI ChatCompletion message.

        :return: OpenAI-compatible message payload.
        """
        ...

    def to_str(self) -> str:
        """
        Return a human-readable string representation of the message.

        :return: Serialized message string.
        """
        ...