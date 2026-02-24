# Class NaiveSearchResult (defined in ragu/search_engine/types.py at lines 88-119)

@dataclasses.dataclass
class NaiveSearchResult:
    """
    Retrieval payload for vector-only (naive) search.
    """
    ...

    chunks: list=field(default_factory=list)

    scores: list=field(default_factory=list)

    documents_id: list[str]=field(default_factory=list)