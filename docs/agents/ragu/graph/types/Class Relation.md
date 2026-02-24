# Class Relation (defined in ragu/graph/types.py at lines 83-118)

@dataclasses.dataclass(slots=True)
class Relation:
    """
    Represents a directed relation between two entities.

    :param subject_id: ID of the source (subject) entity.
    :param object_id: ID of the target (object) entity.
    :param subject_name: Display name of the subject entity.
    :param object_name: Display name of the target entity.
    :param description: Description of the relationship.
    :param relation_strength: Numerical weight of the relation (default: 1.0).
    :param source_chunk_id: Identifiers of chunks where this relation was extracted.
    :param id: Unique identifier; auto-generated if not provided.
    """
    ...

    subject_id: str

    object_id: str

    subject_name: str

    object_name: str

    relation_type: str

    description: str

    relation_strength: int | float = 1.0

    source_chunk_id: list[str] = field(default_factory=list)

    id: str | None = None