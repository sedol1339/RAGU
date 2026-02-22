from abc import abstractmethod, ABC

from ragu.utils.ragu_utils import FLOATS


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders.
    """

    def __init__(self, dim: int | None = None):
        """
        Initialize base embedder.

        :param dim: Embedding dimensionality if known.
        """
        self.dim = dim

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]] | FLOATS:
        """
        Computes embeddings for a list of text inputs.

        :param texts: Input texts.
        :return: Embedding vectors aligned with input order.
        """
        ...

    async def embed_single(self, text: str) -> list[float] | FLOATS:
        """
        Computes embeddings for a single text input.
        """
        return (await self.embed([text]))[0]
