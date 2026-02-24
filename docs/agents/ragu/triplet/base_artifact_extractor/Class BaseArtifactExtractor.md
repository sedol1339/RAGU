# Class BaseArtifactExtractor (defined in ragu/triplet/base_artifact_extractor.py at lines 11-66)

class BaseArtifactExtractor(ragu.common.base.RaguGenerativeModule, abc.ABC):
    """
    Abstract base class for entity and relation extraction modules.

    This class defines a unified interface for all artifact extraction components
    used in the RAG pipeline. Subclasses must implement the :meth:`extract`
    method to transform raw text chunks into structured graph entities and relations.
    """
    ...

    @abc.abstractmethod
    async def extract(
        self,
        chunks: typing.Iterable[ragu.chunker.types.Chunk],
        *args,
        **kwargs
    ) -> typing.Tuple[typing.List[ragu.graph.types.Entity], typing.List[ragu.graph.types.Relation]]:
        """
        Abstract method for extracting entities and relations from text chunks.

        Subclasses must implement this method and return all extracted entities
        and relations corresponding to the provided text inputs.

        :param chunks: Iterable of :class:`Chunk` objects containing text content.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: A tuple ``(entities, relations)`` with lists of extracted objects.
        """
        ...