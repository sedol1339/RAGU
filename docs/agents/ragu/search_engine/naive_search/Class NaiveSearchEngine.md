# Class NaiveSearchEngine (defined in ragu/search_engine/naive_search.py at lines 20-167)

class NaiveSearchEngine(ragu.search_engine.base_engine.BaseEngine):
    """
    Performs naive vector RAG search over document chunks.

    This engine retrieves chunks most similar to a query using vector embeddings,
    optionally reranks them, and passes the context to an LLM for response generation.
    """
    ...

    async def a_search(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_k: typing.Optional[int] = None,
        *args,
        **kwargs
    ) -> ragu.search_engine.types.NaiveSearchResult:
        """
        Perform a naive vector search over chunks.

        :param query: Input query string.
        :param top_k: Number of top chunks to retrieve initially.
        :param rerank_top_k: Number of chunks to keep after reranking.
                             If None, keeps all reranked chunks. Used only when reranker is set.
        :return: NaiveSearchResult with retrieved chunks, scores, and document ids.
        """
        ...

    async def a_query(self, query: str, top_k: int = 20, rerank_top_k: typing.Optional[int] = None) -> str | pydantic.BaseModel:
        """
        Execute a retrieval-augmented query using naive vector search.

        :param query: User query in natural language.
        :param top_k: Number of chunks to search initially (default: 20).
        :param rerank_top_k: Number of chunks to use after reranking (default: None = use all).
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        :rtype: str | BaseModel
        """
        ...