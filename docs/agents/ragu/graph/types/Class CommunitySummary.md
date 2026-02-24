# Class CommunitySummary (defined in ragu/graph/types.py at lines 163-173)

@dataclasses.dataclass(slots=True)
class CommunitySummary:
    """
    Stores a textual summary of a community.

    :param id: Unique identifier of the community.
    :param summary: Generated textual summary of the community content.
    """
    ...

    id: str

    summary: str