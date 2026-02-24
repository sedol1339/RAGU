# Class JsonKVStorage (defined in ragu/storage/kv_storage_adapters/json_storage.py at lines 10-125)

class JsonKVStorage(ragu.storage.base_storage.BaseKVStorage):
    """
    Key-value storage implementation using a local JSON file.

    This class provides a simple persistent storage backend.
    All data is loaded into memory at initialization and written back to disk upon updates.
    """
    ...

    async def all_keys(self) -> list[str]:
        """
        Return a list of all keys currently stored in the JSON file.

        :return: List of keys present in the store.
        """
        ...

    async def get_by_id(self, id):
        """
        Retrieve a record by its unique identifier.

        :param id: Unique identifier key.
        :return: The stored value or ``None`` if not found.
        """
        ...

    async def get_by_ids(self, ids, fields=None):
        """
        Retrieve multiple records by their identifiers.

        Optionally return only specified fields from each record.

        :param ids: Iterable of record IDs to fetch.
        :param fields: Optional list of fields to include in the result.
        :return: List of records or field-filtered dictionaries.
        """
        ...

    async def filter_keys(self, data: list[str]) -> set[str]:
        """
        Return a subset of keys that are not yet present in the store.

        :param data: List of keys to check.
        :return: Set of keys missing from the storage.
        """
        ...

    async def upsert(self, data: dict[str, dict]):
        """
        Insert or update one or more key-value pairs in the store.

        :param data: Dictionary of key-value mappings to update.
        """
        ...

    async def delete(self, ids: list[str]) -> None:
        """
        Delete multiple records by their IDs from the key-value store.

        :param ids: List of IDs to remove.
        :type ids: list[str]
        """
        ...

    async def drop(self):
        """
        Remove all records from the store (in-memory only).
        """
        ...

    async def index_start_callback(self):
        """
        Pre-index hook for interface compatibility.
        """
        ...

    async def query_done_callback(self):
        """
        Post-query hook for interface compatibility.
        """
        ...

    async def index_done_callback(self):
        """
        Persist the current in-memory data to disk.
        """
        ...