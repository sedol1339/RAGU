# Class ChunkContext (defined in ragu/triplet/ragu_lm_artifact_extractor.py at lines 25-35)

@dataclasses.dataclass
class ChunkContext:
    """
    Tracks extraction state for a single chunk through all pipeline stages.
    """
    ...

    chunk: ragu.chunker.types.Chunk

    raw_entities: typing.List[str] = field(default_factory=list)

    normalized_entities: typing.List[str] = field(default_factory=list)

    entities: typing.List[ragu.graph.types.Entity] = field(default_factory=list)

    relations: typing.List[ragu.graph.types.Relation] = field(default_factory=list)