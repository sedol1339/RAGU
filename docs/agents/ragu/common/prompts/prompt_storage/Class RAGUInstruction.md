# Class RAGUInstruction (defined in ragu/common/prompts/prompt_storage.py at lines 45-50)

@dataclasses.dataclass(frozen=True, slots=True)
class RAGUInstruction:
...

    messages: ragu.common.prompts.messages.ChatMessages

    pydantic_model: typing.Optional[typing.Type[pydantic.BaseModel]] = None

    description: typing.Optional[str] = None