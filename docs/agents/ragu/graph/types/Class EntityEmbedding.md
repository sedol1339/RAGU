# Class EntityEmbedding (defined in ragu/graph/types.py at lines 69-81)

@dataclasses.dataclass(slots=True)
class EntityEmbedding:
    """
    Stores vector embeddings for a single entity.

    :param id: ID corresponding to the associated :class:`Entity`.
    :param name_embedding: Embedding vector for the entity name.
    :param description_embedding: Embedding vector for the entity description.
    """
    ...

    id: str

    name_embedding: np.ndarray | None = None

    description_embedding: np.ndarray | None = None,