# Class Community (defined in ragu/graph/types.py at lines 132-161)

@dataclasses.dataclass(slots=True)
class Community:
    """
    Represents a detected community or cluster in the knowledge graph.

    :param level: Hierarchical clustering level of the community.
    :param cluster_id: Identifier of the cluster within the given level.
    :param entities: List of :class:`Entity` objects belonging to the community.
    :param relations: List of :class:`Relation` objects connecting entities.
    :param id: Unique identifier; auto-generated if not provided.
    """
    ...

    level: int

    cluster_id: int

    entities: typing.List[ragu.graph.types.Entity]

    relations: typing.List[ragu.graph.types.Relation]

    id: str | None = None