# Class PendingRequest (defined in ragu/common/cache.py at lines 19-27)

@dataclasses.dataclass(frozen=True, slots=True)
class PendingRequest:
    """
    Represents a request pending generation (not found in cache).
    """
    ...

    index: int

    messages: ragu.common.prompts.ChatMessages

    cache_key: str