# Class CachedOpenAI (defined in ragu/llm/llm.py at lines 176-350)

@dataclasses.dataclass
class CachedOpenAI(ragu.llm.llm.CachedLLM):
    """OpenAI API implementation that enables structured outputs and
    embeddings, response caching, rate limiting and request retrying.

    If `client` is provided, the arguments `base_url` and `api_key`
    are not used. Otherwise, a new `AsyncOpenAI` client is constructed.

    ### Schema handling

    If `output_schema == str`, runs `client.chat.completions.create`
    and returns the `response.choices[0].message.content`.

    If `output_schema != str`, then an additional parameter `as_tool`
    offers two different ways to handle the `output_schema`. The
    correctness and quality of the responses is model-dependent and
    provider-dependent:

    - If `as_tool=True`: calls `client.chat.completions.create` and
      passed `tool_definition` that contain the output format schema.
    - If `as_tool=False`: calls `client.beta.chat.completions.parse` and
      passed the `response_format` argument.

    ### Rate limits and retrying

    Rates can be controlled by:
    - `rate_min_delay`: min delay in seconds between requests
    - `rate_max_per_minute`: max requests per minute
    - `rate_max_simultaneous`: max simultaneous requests

    Allows retrying: for example, if `retry_times=(4, 8, 16)`, will
    retry in 4, then 8, then 16 seconds on exception, and finally
    raise it. In rate limiting, each retrying attempt is considered
    a new request.

    So, these mechanisms are independent: rate limiting delays
    requests, and retrying handles exceptions.

    ### Response caching

    Typically, pass `cache="my_cache_dir/"` to enable caching. For
    details see `StructuredOutputLLM`.
    """
    ...