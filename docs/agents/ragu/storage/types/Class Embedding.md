# Class Embedding (defined in ragu/storage/types.py at lines 10-27)

@dataclasses.dataclass(slots=True)
class Embedding:
    """
    Representation of an embedding.

    :param id: Unique record identifier.
    :param vector: Embedding vector.
    :param metadata: Additional payload.
    """
    ...

    vector: typing.List[float] | np.ndarray

    metadata: typing.Dict[str, typing.Any] = field(default_factory=dict)

    id: typing.Optional[str] = None