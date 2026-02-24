# Class CachedLLM (defined in ragu/llm/llm.py at lines 25-172)

class CachedLLM:
    """An abstract LLM able to respond with texts, structured schemas
    or embeddings.

    Is made to unify backends (openai, pydantic_ai, instructor etc.),
    primarily to enable backend-agnostic response caching.

    ### How caching works

    Uses abstract dict (str -> Any) as cache, typically this may
    be a dict() for in-memory caching, or diskcache.Index for disk
    caching.

    Caching key is calculated by combining `chat_completion` or
    `embed_text` arguments and `cache_prefix`.

    ### Subclassing rules

    1. Override `_chat_completion` and/or `_embed_text` in subclass,
       while `chat_completion` and `embed_text` in base class serve as
       a caching wrapper.
    2. Call `super().__init__(cache, prefix)` in constructor if you
       need to enable caching.
    3. Optionally may add more keyword arguments to `chat_completion`,
       such as `temperature`, `tools` etc, they will also be added in
       the caching key calculation.
    4. If you have object-level parameters, such as `temperature`,
       consider moving them into `chat_completion` arguments, so that
       temperature value is cached cofrrectly, or add them as `cache_prefix`.
       The `cache_prefix` may also be used if the same cache is reused by
       multiple `StructuredOutputLLM` subclasses that return different
       results for the same input parameters in `chat_completion`.
    """
    ...

    cache: collections.abc.MutableMapping[str, typing.Any] | None = None

    async def chat_completion(  # with caching
        self,
        model_name: str,
        conversation: list[openai.types.chat.ChatCompletionMessageParam],
        output_schema: type[ragu.llm.llm.T] = str,
        **kwargs: typing.Any,
    ) -> ragu.llm.llm.T:
    ...

    async def embed_text(  # with caching
        self,
        model_name: str,
        text: str,
        **kwargs: typing.Any,
    ) -> list[float] | ragu.utils.ragu_utils.FLOATS:
    ...