# Function compute_mdhash_id (defined in ragu/utils/ragu_utils.py at lines 106-118)

def compute_mdhash_id(*args: str, prefix: str = '', **kwargs: str) -> str:
    """A unique string hash for the given combination of arguments.
    Invariant to kwargs order.
    """
    ...