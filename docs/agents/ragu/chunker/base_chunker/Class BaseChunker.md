# Class BaseChunker (defined in ragu/chunker/base_chunker.py at lines 6-34)

class BaseChunker(abc.ABC):
    """
    Abstract base class for text chunking strategies.
    Should be subclassed with specific chunking implementations.
    """
    ...

    @abc.abstractmethod
    def split(self, documents: str | typing.Sequence[str]) -> typing.List[ragu.chunker.types.Chunk]:
        """
        Abstract method for splitting documents into smaller chunks.
        Must be implemented in subclasses.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        ...