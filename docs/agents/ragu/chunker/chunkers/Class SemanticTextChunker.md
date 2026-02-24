# Class SemanticTextChunker (defined in ragu/chunker/chunkers.py at lines 73-180)

class SemanticTextChunker(ragu.chunker.base_chunker.BaseChunker):
    """
    A semantic chunker that splits text into coherent chunks
    based on sentence boundaries and semantic similarity.
    """
    ...

    def split(self, documents: str | typing.List[str]) -> typing.List[ragu.chunker.types.Chunk]:
        """
        Splits input documents into semantically coherent chunks.

        :param documents: A single document or a list of documents.
        :return: List of `Chunk` objects with content and IDs.
        """
        ...