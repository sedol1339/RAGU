# Class BaseKVStorage (defined in ragu/storage/base_storage.py at lines 77-147)

@dataclasses.dataclass
class BaseKVStorage(typing.Generic[ragu.storage.base_storage.T], ragu.storage.base_storage.BaseStorage, abc.ABC):
    """
    Abstract interface for key-value storage backends.
    """
    ...

    @abc.abstractmethod
    async def all_keys(self) -> typing.List[str]:
        """
        Return all currently stored keys.

        :return: List of key strings.
        """
        ...

    @abc.abstractmethod
    async def get_by_id(self, id: str) -> typing.Union[ragu.storage.base_storage.T, None]:
        """
        Fetch one value by key.

        :param id: Key to retrieve.
        :return: Stored value or ``None`` if absent.
        """
        ...

    @abc.abstractmethod
    async def get_by_ids(self, ids: list[str], fields: typing.Union[set[str], None] = None) -> typing.List[typing.Union[ragu.storage.base_storage.T, None]]:
        """
        Fetch multiple values by key.

        :param ids: Keys to retrieve in order.
        :param fields: Optional field projection for dict-like values.
        :return: Values aligned with ``ids``; missing keys mapped to ``None``.
        """
        ...

    @abc.abstractmethod
    async def filter_keys(self, data: typing.List[str]) -> typing.Set[str]:
        """
        Return keys from input that do not exist in storage.

        :param data: Candidate keys.
        :return: Subset of keys that are currently missing.
        """
        ...

    @abc.abstractmethod
    async def upsert(self, data: typing.Dict[str, ragu.storage.base_storage.T]):
        """
        Insert or update key-value entries.

        :param data: Mapping of keys to values.
        """
        ...

    @abc.abstractmethod
    async def delete(self, ids: typing.List[str]) -> None:
        """
        Delete entries by keys.

        :param ids: Keys to delete.
        """
        ...

    @abc.abstractmethod
    async def drop(self):
        """
        Remove all entries from the storage backend.
        """
        ...