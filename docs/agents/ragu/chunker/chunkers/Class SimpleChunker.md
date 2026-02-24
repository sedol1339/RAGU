# Class SimpleChunker (defined in ragu/chunker/chunkers.py at lines 13-71)

class SimpleChunker(ragu.chunker.base_chunker.BaseChunker):
    """
    A simple chunker that splits text into fixed-size overlapping chunks.
    """
    ...

    def split(self, documents: str | typing.List[str]) -> typing.List[ragu.chunker.types.Chunk]:
        """
        Splits documents into fixed-size overlapping chunks.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        ...