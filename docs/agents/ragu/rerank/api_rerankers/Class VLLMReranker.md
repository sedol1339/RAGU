# Class VLLMReranker (defined in ragu/rerank/api_rerankers.py at lines 12-95)

class VLLMReranker(ragu.rerank.base_reranker.BaseReranker):
    """
    Reranker that uses vLLM's /v1/score endpoint.
    Compatible with vLLM serve running cross-encoder models.
    """
    ...

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=1, max=8))
    async def rerank(
            self,
            x: str,
            others: typing.List[str],
            top_k: int | None = None,
    ) -> typing.List[typing.Tuple[int, float]]:
        """
        Reranks documents based on relevance to the query.

        :param x: Query text.
        :param others: List of documents/items to rerank.
        :param top_k: Number of top results to return. If None, returns all.
        :return: List of (index, score) tuples sorted by relevance descending.
        """
        ...

    async def aclose(self):
        """
        Close underlying HTTP client.
        """
        ...