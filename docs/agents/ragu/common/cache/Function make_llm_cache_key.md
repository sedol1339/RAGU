# Function make_llm_cache_key (defined in ragu/common/cache.py at lines 30-62)

def make_llm_cache_key(
    content: str,
    model_name: typing.Optional[str] = None,
    schema: typing.Optional[typing.Type[pydantic.BaseModel]] = None,
    kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> str:
    """
    Build a deterministic cache key from LLM request parameters.

    :param model_name: Model name used for generation.
    :param schema: Optional Pydantic schema class.
    :param kwargs: Additional API parameters.
    :return: A unique cache key string.
    """
    ...