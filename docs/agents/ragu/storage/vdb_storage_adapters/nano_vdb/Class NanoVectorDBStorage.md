# Class NanoVectorDBStorage (defined in ragu/storage/vdb_storage_adapters/nano_vdb.py at lines 15-148)

class NanoVectorDBStorage(ragu.storage.base_storage.BaseVectorStorage):
    """
    Vector storage implementation using NanoVectorDB as the backend.

    This class provides a simple vector database for storing and retrieving
    embeddings, enabling similarity search operations such as nearest
    neighbor queries.
    """
    ...

    async def upsert(self, data: typing.List[ragu.storage.types.Embedding]) -> typing.List[typing.Any]:
        """
        Insert or update a batch of embeddings in the database.

        :param data: Embedding records with vectors and metadata.
        :return: List of records successfully inserted or updated.
        """
        ...

    async def query(self, vector: ragu.storage.types.Embedding, top_k: int = 5) -> typing.List[ragu.storage.types.EmbeddingHit]:
        """
        Search for the most similar documents in the vector database.

        Performs a cosine similarity search against all stored vectors,
        returning the top ``k`` results exceeding the similarity threshold.

        :param vector: Query embedding payload.
        :param top_k: Number of nearest neighbors to return.
        :return: List of matched records and their distances.
        """
        ...

    async def index_start_callback(self):
        """
        Pre-index hook for interface compatibility.
        """
        ...

    async def query_done_callback(self):
        """
        Post-query hook for interface compatibility.
        """
        ...

    async def delete(self, ids: typing.List[str]) -> None:
        """
        Delete embeddings by their IDs from the vector database.

        :param ids: List of IDs to remove from the vector storage.
        :type ids: List[str]
        """
        ...

    async def index_done_callback(self) -> None:
        """
        Save the current state of the NanoVectorDB to disk.

        This method ensures that any newly inserted or updated vectors
        are persisted in the storage file.
        """
        ...