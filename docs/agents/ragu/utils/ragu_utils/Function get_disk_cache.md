# Function get_disk_cache (defined in ragu/utils/ragu_utils.py at lines 28-37)

def get_disk_cache(dir: str | pathlib.Path) -> collections.abc.MutableMapping[str, typing.Any]:
    """Get or create a DiskCache by a directory name.
    Cache is shared between multiple `get_disk_cache` calls.
    """
    ...