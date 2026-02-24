# Class StorageArguments (defined in ragu/storage/index.py at lines 40-63)

@dataclasses.dataclass
class StorageArguments:
    """
    Configuration for Index storage backends.

    :param graph_backend_storage: Storage backend class for graph structure (nodes/edges).
    :param kv_storage_type: Storage backend class for key-value data (chunks, communities, summaries).
    :param vdb_storage_type: Storage backend class for vector embeddings (entities, relations, chunks).
    :param chunks_kv_storage_kwargs: Additional kwargs passed to KV storage for text chunks.
    :param summary_kv_storage_kwargs: Additional kwargs passed to KV storage for community summaries.
    :param communities_kv_storage_kwargs: Additional kwargs passed to KV storage for community metadata.
    :param vdb_storage_kwargs: Additional kwargs passed to vector database instances.
    :param graph_storage_kwargs: Additional kwargs passed to graph backend storage.
    """
    ...

    graph_backend_storage: typing.Type[ragu.storage.base_storage.BaseGraphStorage] = NetworkXStorage

    kv_storage_type: typing.Type[ragu.storage.base_storage.BaseKVStorage] = JsonKVStorage

    vdb_storage_type: typing.Type[ragu.storage.base_storage.BaseVectorStorage] = NanoVectorDBStorage

    chunks_kv_storage_kwargs: typing.Dict = field(default_factory=dict)

    summary_kv_storage_kwargs: typing.Dict = field(default_factory=dict)

    communities_kv_storage_kwargs: typing.Dict = field(default_factory=dict)

    vdb_storage_kwargs: typing.Dict = field(default_factory=dict)

    graph_storage_kwargs: typing.Dict = field(default_factory=dict)