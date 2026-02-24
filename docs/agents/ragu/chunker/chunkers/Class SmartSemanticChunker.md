# Class SmartSemanticChunker (defined in ragu/chunker/chunkers.py at lines 182-254)

class SmartSemanticChunker(ragu.chunker.base_chunker.BaseChunker):
    """
    A smart semantic chunker using a reranker-based algorithm to prepare a long document for retrieval augmented generation.

    For more information, see https://github.com/bond005/smart_chunker/tree/main
    """
    ...

    def split(self, documents: str | typing.List[str]) -> typing.List[ragu.chunker.types.Chunk]:
        """
        Splits documents using the SmartChunker model.

        :param documents: A single document or a list of documents.
        :return: List of `Chunk` objects with per-document indexing.
        """
        ...