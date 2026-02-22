import asyncio
from collections.abc import Awaitable, Collection, MutableMapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack
import functools
from hashlib import md5
from pathlib import Path
from typing import Callable, Any, TypeVar, cast
from typing import List

from diskcache import Index # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
from aiolimiter import AsyncLimiter

from ragu.common.logger import logger


FLOATS = npt.NDArray[np.floating[Any]]
"""A typization for numpy array of floats"""

INTS = npt.NDArray[np.integer[Any]]
"""A typization for numpy array of integers"""

_dish_caches: dict[str, Index] = {}

def get_disk_cache(dir: str | Path) -> MutableMapping[str, Any]:
    """Get or create a DiskCache by a directory name.
    Cache is shared between multiple `get_disk_cache` calls.
    """
    path = str(Path(dir).resolve())
    if (cache := _dish_caches.get(path, None)):
        return cache
    _dish_caches[path] = cache = Index(path)
    return cache


T_fn = TypeVar('T_fn', bound=Callable[..., Awaitable[Any]])

def attach_async_contexts(
    func: T_fn,
    *contexts: AbstractAsyncContextManager[Any],
) -> T_fn:
    """Wraps the `func` into the given async contexts."""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug('attach_async_contexts: entering context!')
        async with AsyncExitStack() as stack:
            for mgr in contexts:
                await stack.enter_async_context(mgr)
            print('attach_async_contexts: entered context!')
            return await func(*args, **kwargs)
            
    return cast(T_fn, wrapper)


class AsyncRunner:
    def __init__(
        self,
        semaphore: asyncio.Semaphore,
        rps_limiter: AsyncLimiter,
        rpm_limiter: AsyncLimiter,
        progress_bar,
    ):
        self.semaphore = semaphore
        self.rps_limiter = rps_limiter
        self.rpm_limiter = rpm_limiter
        self.progress_bar = progress_bar

    async def make_request(self, func: Callable[..., Any], **kwargs):
        async with self.semaphore:
            async with self.rps_limiter:
                async with self.rpm_limiter:
                    try:
                        return await func(**kwargs)
                    finally:
                        self.progress_bar.update(1)


def compute_mdhash_id(*args: str, prefix: str = '', **kwargs: str) -> str:
    """A unique string hash for the given combination of arguments.
    Invariant to kwargs order.
    """
    string = ''
    for x in args:
        assert isinstance(x, str)
        string += '\0' + x
    for key, x in sorted(kwargs.items(), key=lambda item: item[0]):
        assert isinstance(x, str)
        string += '\0' + key + '\1' + x
    return prefix + md5(string.encode()).hexdigest()


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError()
        return current_loop

    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


def read_text_from_files(directory: str | Path, file_extensions: Collection[str] | None = None) -> List[str]:
    texts = []
    directory = Path(directory)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and (file_extensions is None or file_path.suffix in file_extensions):
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    texts.append(f.read())
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"⚠️ Cannot read file {file_path}: {e}")

    return texts

