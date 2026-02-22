import asyncio
import json
import logging
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, TypeVar, cast
from typing_extensions import override

from pydantic import BaseModel
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed, before_sleep_log
from aiolimiter import AsyncLimiter

from ragu.utils.ragu_utils import FLOATS, LoguruAdapter, attach_async_contexts, get_disk_cache
from ragu.common.logger import logger


# LLM Interfaces

T = TypeVar('T', BaseModel, str)

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

    cache: MutableMapping[str, Any] | None = None

    def __init__(
        self,
        cache: MutableMapping[str, Any] | str | Path | None = None,
        cache_prefix: str = '',
    ):
        self.cache_prefix = cache_prefix
        match cache:
            case None:
                self.cache = None
            case str() | Path():
                self.cache = get_disk_cache(cache)
            case _:
                self.cache = cache

    async def chat_completion(  # with caching
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        is_str = issubclass(output_schema, str)
        args: dict[str, Any] = {
            'cache_prefix': self.cache_prefix,
            'model_name': model_name,
            'method': 'chat_completion',
            'conversation': conversation,
            'output_schema': 'str' if is_str else output_schema.model_json_schema(),
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if self.cache is not None and (value := self.cache.get(key, None)):
            logger.debug(f'Cache hit for {model_name}! Returning from cache.')
            cached: str | dict[str, Any]
            _args, cached = value
            result = cached if is_str else output_schema.model_validate(cached)
            return cast(T, result)
        
        if self.cache is not None:
            logger.debug(f'Cache miss for {model_name}! Doing a request.')
        
        response = await self._chat_completion(
            model_name=model_name,
            conversation=conversation,
            output_schema=output_schema,
            **kwargs,
        )

        cached = response if is_str else response.model_dump() # type: ignore

        if self.cache is not None:
            self.cache[key] = args, cached

        return response

    async def _chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        # should be overridden only if a subcclass supports chat completions
        # kwargs are here to add custom arguments that will also be cached
        raise NotImplementedError()
    
    async def embed_text(  # with caching
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        args: dict[str, Any] = {
            'cache_prefix': self.cache_prefix,
            'model_name': model_name,
            'method': 'embed_text',
            'text': text,
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if self.cache is not None and (value := self.cache.get(key, None)):
            logger.debug(f'Cache hit for {model_name}! Returning from cache.')
            cached: list[float] | FLOATS
            _args, cached = value
            return cached
        
        if self.cache is not None:
            logger.debug(f'Cache miss for {model_name}! Doing a request.')
        
        response = await self._embed_text(
            model_name=model_name,
            text=text,
            **kwargs,
        )

        if self.cache is not None:
            self.cache[key] = args, response

        return response

    async def _embed_text(
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        # should be overridden only if a subcclass supports embeddings
        # kwargs are here to add custom arguments that will also be cached
        raise NotImplementedError()


# LLM Implementations

@dataclass
class CachedOpenAI(CachedLLM):
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

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        rate_min_delay: float | None = None,
        rate_max_per_minute: int | None = None,
        rate_max_simultaneous: int | None = None,
        retry_times_sec: Sequence[float] | None = None,
        cache: MutableMapping[str, Any] | str | Path | None = None,
        cache_prefix: str = 'openai',
    ):
        super().__init__(cache=cache, cache_prefix=cache_prefix)

        self.client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Should add retrying after attaching limiters, so that
        # every retry increments counter in the limiters.

        # Thus, handlers/wrappers will be called in this order:
        # 1. Caching
        # 2. Retrying
        # 3. Rate limiting

        # add rate limiter contexts
        contexts: list[AbstractAsyncContextManager[Any]] = []
        if rate_max_per_minute:
            contexts.append(AsyncLimiter(rate_max_per_minute, time_period=60))
        if rate_max_simultaneous:
            contexts.append(asyncio.Semaphore(rate_max_simultaneous))
        if rate_min_delay:
            contexts.append(AsyncLimiter(1, time_period=rate_min_delay))
        if contexts:
            self._chat_completion = attach_async_contexts(
                self._chat_completion, *contexts
            )
            self._embed_text = attach_async_contexts(
                self._embed_text, *contexts
            )

        # add retrying decorators
        if retry_times_sec:
            retrying_decorator = retry(
                stop=stop_after_attempt(len(retry_times_sec) + 1),
                wait=wait_chain(*[wait_fixed(t) for t in retry_times_sec]),
                before_sleep=before_sleep_log(
                    LoguruAdapter('logger'), logging.DEBUG
                ),
                reraise=True
            )
            self._chat_completion = retrying_decorator(self._chat_completion)
            self._embed_text = retrying_decorator(self._embed_text)

    @override
    async def _chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        as_tool: bool = False,
        **kwargs: Any,
    ) -> T:
        logger.debug(f'Sending chat_completion API request...')
        if issubclass(output_schema, str):
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=conversation,
            )
            content = response.choices[0].message.content
            return cast(T, content if content is not None else '')

        model_schema = output_schema
        
        if not as_tool:
            # use response_format
            parsed_completion = await self.client.beta.chat.completions.parse(
                model=model_name,
                messages=conversation,
                response_format=model_schema, 
            )
            
            parsed_result = parsed_completion.choices[0].message.parsed
            
            if parsed_result is None:
                raise ValueError('OpenAI refused to output structured data.')
            return cast(T, parsed_result)

        else:
            # use tool calling to define schema, as in pydantic_ai
            function_name = model_schema.__name__
            tool_definition: ChatCompletionFunctionToolParam = {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": f"Output data in the structure of {function_name}",
                    "parameters": model_schema.model_json_schema(), # type: ignore
                },
            }

            response = await self.client.chat.completions.create(
                model=model_name,
                messages=conversation,
                tools=[tool_definition],
                tool_choice={"type": "function", "function": {"name": function_name}},
            )

            message = response.choices[0].message
            
            if not message.tool_calls:
                raise ValueError('Model did not call the expected tool.')
            
            # Parse the arguments from the tool call back into the Pydantic model
            arguments_json = cast(str, message.tool_calls[0].function.arguments) # type: ignore
            return cast(T, model_schema.model_validate_json(arguments_json))

    @override
    async def _embed_text(
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        logger.debug(f'Sending embed_text API request...')
        response = await self.client.embeddings.create(
            model=model_name, input=text,
        )
        return response.data[0].embedding