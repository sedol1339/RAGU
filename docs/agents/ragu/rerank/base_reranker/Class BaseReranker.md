# Class BaseReranker (defined in ragu/rerank/base_reranker.py at lines 10-79)

class BaseReranker(abc.ABC):
    """
    Base interface for document rerankers.

    Subclasses score candidate documents relative to a query and return
    sorted (index, score) pairs.
    """
    ...

    @abc.abstractmethod
    async def rerank(self, x: str, others: typing.List[str], **kwargs) -> typing.List[typing.Tuple[int, float]]:
        """
        Score and rank candidate texts for a single query.

        :param x: Query text.
        :param others: Candidate documents.
        :param kwargs: Provider-specific scoring parameters.
        :return: Ranked ``(index, score)`` tuples in descending order.
        """
        ...

    async def batch_rerank(
            self,
            queries: typing.List[str],
            documents: typing.List[typing.List[str]],
            progress_bar_desc: str | None = None,
            **kwargs
    ) -> typing.List[typing.List[typing.Tuple[int, float]]]:
        """
        Reranks multiple query-documents pairs in batch with rate limiting.

        :param queries: List of query texts.
        :param documents: List of document lists, one per query.
        :param progress_bar_desc: Description for progress bar.
        :param kwargs: Additional arguments passed to rerank.
        :return: List of rerank results for each query.
        """
        ...