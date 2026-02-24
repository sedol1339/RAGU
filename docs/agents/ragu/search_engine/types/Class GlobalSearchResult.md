# Class GlobalSearchResult (defined in ragu/search_engine/types.py at lines 63-86)

@dataclasses.dataclass
class GlobalSearchResult:
    """
    Aggregated global-search insights with relevance ratings.
    """
    ...

    insights: list=field(default_factory=list)