# Class RelationEmbedding (defined in ragu/graph/types.py at lines 120-130)

@dataclasses.dataclass(slots=True)
class RelationEmbedding:
    """
    Stores a vector embedding for a relation.

    :param id: ID corresponding to the associated :class:`Relation`.
    :param embedding: Embedding vector capturing the relation’s semantic meaning.
    """
    ...

    id: str

    embedding: np.ndarray | None = None,