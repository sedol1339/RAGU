# Class Entity (defined in ragu/graph/types.py at lines 32-67)

@dataclasses.dataclass(slots=True)
class Entity:
    """
    Represents a semantic entity (graph node).

    :param entity_name: Canonical name of the entity.
    :param entity_type: Type or category of the entity (e.g., Person, Organization).
    :param description: Textual description extracted from the source text.
    :param source_chunk_id: Identifiers of text chunks where the entity was found.
    :param documents_id: Identifiers of documents containing this entity.
    :param clusters: List of cluster memberships from community detection.
    :param id: Unique identifier; auto-generated if not provided.
    """
    ...

    entity_name: str

    entity_type: str

    description: str

    source_chunk_id: list[str]

    documents_id: list[str] = field(default_factory=list)

    clusters: list[dict] = field(default_factory=list)

    id: str | None = None