# Class CrossEncoderReranker (defined in ragu/rerank/local_rerankers.py at lines 7-62)

class CrossEncoderReranker(ragu.rerank.base_reranker.BaseReranker):
    """
    Reranker that uses Sentence Transformers CrossEncoder to compute relevance scores.
    """
    ...

    async def rerank(
            self,
            x: str,
            others: typing.List[str],
            batch_size: int = 16,
            top_k: int | None = None
    ) -> typing.List[typing.Tuple[int, float]]:
        """
        Reranks documents based on relevance to the query.

        :param x: Query text.
        :param others: List of documents to rerank.
        :param batch_size: Batch size for inference.
        :param top_k: Number of top results to return. If None, returns all.
        :return: List of (index, score) tuples sorted by relevance descending.
        """
        ...