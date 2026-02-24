# Class EmbeddingCache (defined in ragu/common/cache.py at lines 232-343)

class EmbeddingCache:
    """
    Async cache specifically for embeddings (lists of floats).

    Uses pickle serialization for efficient storage of numeric data.
    Separate from LLM cache to allow independent management and different storage strategies.
    """
    ...

    async def flush_cache(self) -> None:
        """
        Flush cache to disk.
        """
        ...

    async def get(self, key: str) -> typing.Optional[typing.List[float]]:
        """
        Retrieve an embedding from cache.

        :param key: Cache key.
        :return: Cached embedding (list of floats), or None if not found.
        """
        ...

    async def set(self, key: str, embedding: typing.List[float]) -> None:
        """
        Store an embedding in cache.

        :param key: Cache key.
        :param embedding: Embedding vector to cache (list of floats).
        """
        ...

    async def close(self) -> None:
        """
        Flush any pending writes and close cache.
        """
        ...