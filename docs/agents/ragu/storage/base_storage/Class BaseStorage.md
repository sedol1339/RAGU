# Class BaseStorage (defined in ragu/storage/base_storage.py at lines 11-37)

@dataclasses.dataclass
class BaseStorage(abc.ABC):
    """
    Base contract for all storage backends used by RAGU.
    """
    ...

    @abc.abstractmethod
    async def index_start_callback(self):
        """
        Execute pre-indexing initialization hook.
        """
        ...

    @abc.abstractmethod
    async def index_done_callback(self):
        """
        Execute post-indexing finalization hook.
        """
        ...

    @abc.abstractmethod
    async def query_done_callback(self):
        """
        Execute post-query cleanup hook.
        """
        ...