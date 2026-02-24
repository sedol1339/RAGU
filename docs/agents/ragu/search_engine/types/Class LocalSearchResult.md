# Class LocalSearchResult (defined in ragu/search_engine/types.py at lines 7-61)

@dataclasses.dataclass
class LocalSearchResult:
    """
    Structured retrieval payload returned by local graph search.
    """
    ...

    entities: list=field(default_factory=list)

    relations: list=field(default_factory=list)

    summaries: list=field(default_factory=list)

    chunks: list=field(default_factory=list)

    documents_id: list[str]=field(default_factory=list)