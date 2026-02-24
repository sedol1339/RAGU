# Function render (defined in ragu/common/prompts/messages.py at lines 142-206)

def render(template_conversation: typing.Union[ragu.common.prompts.messages.BaseMessage, ragu.common.prompts.messages.ChatMessages], **params: typing.Any) -> typing.List[ragu.common.prompts.messages.ChatMessages]:
    """
    Render Jinja2 templates inside message contents in batch mode.

    :param template_conversation: Message or conversation template.
    :param params: Scalar and batch Jinja context parameters.
    :return: Rendered conversations (batch size inferred from list/tuple params).
    :raises ValueError: If batch parameter lengths are inconsistent.
    """
    ...