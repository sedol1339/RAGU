# Function attach_async_contexts (defined in ragu/utils/ragu_utils.py at lines 41-56)

def attach_async_contexts(
    func: ragu.utils.ragu_utils.T_fn,
    *contexts: contextlib.AbstractAsyncContextManager[typing.Any],
) -> ragu.utils.ragu_utils.T_fn:
    """Wraps the `func` into the given async contexts."""
    ...