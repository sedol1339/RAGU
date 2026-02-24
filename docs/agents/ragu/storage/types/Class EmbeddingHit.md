# Class EmbeddingHit (defined in ragu/storage/types.py at lines 29-41)

@dataclasses.dataclass(slots=True)
class EmbeddingHit:
    """
    Vector query hit.

    :param id: Matched record identifier.
    :param distance: Similarity/distance score to query embedding.
    :param metadata: Additional payload.
    """
    ...

    id: str

    distance: float

    metadata: typing.Dict[str, typing.Any] = field(default_factory=dict)