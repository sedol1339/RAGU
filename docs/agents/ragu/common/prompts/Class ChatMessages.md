# Class ChatMessages (defined in ragu/common/prompts/messages.py at lines 100-140)

@dataclasses.dataclass(frozen=True, slots=True)
class ChatMessages:
    """
    Container for a list of chat messages.

    Represents a single user-assistant conversation.
    """
    ...

    messages: typing.List[ragu.common.prompts.messages.BaseMessage]

    @classmethod
    def from_messages(cls: typing.Type[ragu.common.prompts.messages.T], messages: typing.Sequence[ragu.common.prompts.messages.BaseMessage]) -> ragu.common.prompts.messages.T:
        """
        Construct a ChatMessages instance from a sequence of messages.

        :param messages: Source message sequence.
        :return: ChatMessages container.
        """
        ...

    def to_openai(self) -> typing.List[openai.types.chat.ChatCompletionMessageParam]:
        """
        Convert all messages to OpenAI ChatCompletion message parameters.

        :return: List of OpenAI-compatible message payloads.
        """
        ...

    def to_str(self) -> str:
        """
        Return a readable multi-line string representation of the conversation.

        :return: Multi-line serialized conversation.
        """
        ...