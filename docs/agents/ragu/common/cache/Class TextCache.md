# Class TextCache (defined in ragu/common/cache.py at lines 70-230)

class TextCache:
    """
    Key-value cache for LLM responses with automatic BaseModel serialization.

    Supports both string and Pydantic BaseModel responses. When a schema is provided,
    automatically serializes/deserializes BaseModel instances.
    """
    ...

    async def flush_cache(self) -> None:
        """
        Flush cache to disk.
        """
        ...

    async def get(
        self,
        key: str,
        *,
        schema: typing.Optional[typing.Type[pydantic.BaseModel]] = None,
    ) -> typing.Optional[typing.Any]:
        """
        Retrieve a value from cache.

        :param key: Cache key.
        :param schema: Optional Pydantic schema to reconstruct BaseModel.
        :return: Cached value (string or BaseModel), or None if not found.
        """
        ...

    async def set(
        self,
        key: str,
        value: typing.Any,
        **additional_payload,
    ) -> None:
        """
        Store a value in cache.

        :param key: Cache key.
        :param value: Value to cache (string or BaseModel).
        """
        ...

    async def close(self) -> None:
        """
        Flush any pending writes and close cache.
        """
        ...