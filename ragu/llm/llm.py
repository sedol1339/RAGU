import asyncio
import json
import logging
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast
from typing_extensions import override

from pydantic import BaseModel
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from tenacity import stop_after_attempt, wait_chain, wait_fixed, Retrying, before_sleep_log
from aiolimiter import AsyncLimiter

from ragu.utils.ragu_utils import attach_async_contexts, get_disk_cache
from ragu.common.logger import logger


# LLM Interfaces

T = TypeVar('T', BaseModel, str)

class StructuredOutputLLM(Protocol):
    """An abstract LLM able to reespond with structured schemas.
    
    Is made to unify backends (openai, pydantic_ai, instructor etc.),
    primarily to enable simple backend-agnostic response caching (see
    `ResponseCached`) and rate limiting (see `StructuredOutputOpenAI`).

    Subclasses may add more keyword arguments to `chat_completion`,
    such as `temperature`, `tools` etc.
    """

    async def chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
        # kwargs is required to add custom arguments that will
        # also be cached in ResponseCached class
    ) -> T: ...

# LLM Implementations

@dataclass
class StructuredOutputOpenAI(StructuredOutputLLM):
    """OpenAI API implementation for `StructuredOutputLLM` interface.

    If `client` is provided, the arguments `base_url` and `api_key`
    are not used. Otherwise, a new `AsyncOpenAI` client is constructed.

    ### Rate limits and retrying
    
    Rates can be controlled by:
    - `rate_min_delay`: min delay in seconds between requests
    - `rate_max_per_minute`: max requests per minute
    - `rate_max_simultaneous`: max simultaneous requests

    Allows retrying: for example, if `retry_times=(4, 8, 16)`, will
    retry in 4, then 8, then 16 seconds on exception, and finally
    raise it. In rate limiting, each retrying attempt is considered
    a new request.

    Thus, these mechanisms are independent: rate limiting delays
    requests, and retrying handles exceptions.

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
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        rate_min_delay: float | None = None,
        rate_max_per_minute: int | None = None,
        rate_max_simultaneous: int | None = None,
        retry_times: Sequence[float] | None = (4, 8, 16),
    ):
        self.client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        self.retry_times = retry_times

        contexts: list[AbstractAsyncContextManager[Any]] = []
        if rate_max_per_minute:
            contexts.append(AsyncLimiter(rate_max_per_minute, time_period=60))
        if rate_max_simultaneous:
            contexts.append(asyncio.Semaphore(rate_max_simultaneous))
        if rate_min_delay:
            contexts.append(AsyncLimiter(1, time_period=rate_min_delay))
        if contexts:
            self.chat_completion = attach_async_contexts(self.chat_completion, *contexts)
    
    @override
    async def chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        as_tool: bool = False,
        **kwargs: Any,
    ) -> T:
        if self.retry_times:
            stop = stop_after_attempt(len(self.retry_times) + 1)
            wait = wait_chain(*[wait_fixed(t) for t in self.retry_times])
        else:
            # disable retrying
            stop = stop_after_attempt(0)
            wait = wait_chain()
        retrying = Retrying(
            stop=stop,
            wait=wait,
            # try to use loguru logger, while `before_sleep_log` expects logging.Logger
            before_sleep=before_sleep_log(logger, logging.DEBUG), # type: ignore
            reraise=True,
        )
        return await retrying(
            self._chat_completion_without_retry,
            model_name=model_name,
            conversation=conversation,
            output_schema=output_schema,
            as_tool=as_tool,
            **kwargs,
        )

    async def _chat_completion_without_retry(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        as_tool: bool = False,
        **kwargs: Any,
    ) -> T:
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

# LLM Wrappers

class ResponseCached(StructuredOutputLLM):
    """A caching wrapper for any `StructuredOutputLLM`.env
    
    Uses abstract dict (str -> Any) as cache, typically this may
    be a dict() for in-memory caching, or diskcache.Index for disk
    caching.

    Caching key is calculated by combining `chat_completion`
    arguments and `cache_prefix`.
    
    So, if you have object-level parameters, such as `temperature`,
    consider moving them into `chat_completion` arguments, so that
    temperature value is cached cofrrectly, or add them as `cache_prefix`.
    The `cache_prefix` may also be used if the same cache is reused by
    multiple `StructuredOutputLLM` subclasses that return different
    results for the same input parameters in `chat_completion`.
    """

    def __init__(
        self,
        model: StructuredOutputLLM,
        cache: MutableMapping[str, Any] | str | Path,
        cache_prefix: str = '',
    ):
        self.model = model
        self.cache = (
            get_disk_cache(cache)
            if isinstance(cache, (str, Path))
            else cache
        )
        self.cache_prefix = cache_prefix

    @override
    async def chat_completion(
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
            'conversation': conversation,
            'output_schema': str if is_str else output_schema.model_json_schema(),
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if (value := self.cache.get(key, None)):
            logger.debug(f'ResponseCached: Cache hit for {model_name}!')
            cached: str | dict[str, Any]
            _args, cached = value
            result = cached if is_str else output_schema.model_validate(cached)
            return cast(T, result)
        
        logger.debug(f'ResponseCached: Cache miss for {model_name}!')
        response = await self.model.chat_completion(
            model_name=model_name,
            conversation=conversation,
            output_schema=output_schema,
        )

        cached = response if is_str else response.model_dump() # type: ignore
        self.cache[key] = args, cached

        return response

