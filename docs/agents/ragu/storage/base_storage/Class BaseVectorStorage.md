# Class BaseVectorStorage (defined in ragu/storage/base_storage.py at lines 39-73)

@dataclasses.dataclass
class BaseVectorStorage(ragu.storage.base_storage.BaseStorage, abc.ABC):
    """
    Abstract interface for vector storage backends.
    """
    ...

    @abc.abstractmethod
    async def query(self, vectors: ragu.storage.types.Embedding, top_k: int) -> typing.List[ragu.storage.types.EmbeddingHit]:
        """
        Retrieve top-k nearest items for a batch of embedding vectors.

        :param vectors: Query embedding vector.
        :param top_k: Maximum number of results to return per query vector.
        :return: A list of query hits with distance score and metadata.
        """
        ...

    @abc.abstractmethod
    async def upsert(self, data: typing.List[ragu.storage.types.Embedding]) -> None:
        """
        Insert or update embedding records.

        :param data: Embedding records to upsert.
        """
        ...

    @abc.abstractmethod
    async def delete(self, ids: typing.List[str]) -> None:
        """
        Delete records by IDs.

        :param ids: Record identifiers to remove.
        """
        ...