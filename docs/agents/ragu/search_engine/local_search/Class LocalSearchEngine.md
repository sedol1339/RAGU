# Class LocalSearchEngine (defined in ragu/search_engine/local_search.py at lines 25-134)

class LocalSearchEngine(ragu.search_engine.base_engine.BaseEngine):
    """
    Performs local retrieval-augmented search (RAG) over a knowledge graph.

    The engine:
      1. Retrieves relevant entities for the query.
      2. Retrieves related items (relations, summary and chunks).
      3. Generates a final response

    Reference
    ---------
    Based on: https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_op.py#L919
    """
    ...

    async def a_search(self, query: str, top_k: int = 20, *args, **kwargs) -> ragu.search_engine.types.LocalSearchResult:
        """
        Retrieve local graph context for the given query.

        :param query: Input query string.
        :param top_k: Number of top entities to retrieve from the entity vector DB.
        :return: LocalSearchResult containing entities, relations, summaries, chunks, and document ids.
        """
        ...

    async def a_query(self, query: str, top_k: int = 20) -> str | pydantic.BaseModel:
        """
        Execute a local RAG query.

        :param query: User query in natural language.
        :param top_k: Number of entities to retrieve into context.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        ...