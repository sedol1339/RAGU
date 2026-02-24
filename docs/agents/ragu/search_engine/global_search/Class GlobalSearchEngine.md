# Class GlobalSearchEngine (defined in ragu/search_engine/global_search.py at lines 18-140)

class GlobalSearchEngine(ragu.search_engine.base_engine.BaseEngine, ragu.common.base.RaguGenerativeModule):
    """
    Executes global retrieval-augmented search (RAG) across the entire knowledge graph.

    Unlike :class:`LocalSearchEngine`, this engine operates at the level of
    *community summaries*, aggregating and ranking high-level semantic clusters
    before generating a global synthesis via the language model.
    """
    ...

    async def a_search(self, query: str, *args, **kwargs) -> ragu.search_engine.types.GlobalSearchResult:
        """
        Perform a global semantic search across all communities in the knowledge graph.

        This method retrieves all available community summaries, sends them to the LLM
        for meta-evaluation, filters out low-rated responses, and returns a ranked
        concatenation of the top relevant community insights.

        :param query: The input natural language query.
        :return: Concatenated responses from the top-rated communities.
        """
        ...

    async def get_meta_responses(self, query: str, context: typing.List[str]) -> typing.List[dict]:
        """
        Generate and evaluate meta-responses for each community summary.

        The model receives the full list of community summaries and scores each
        according to relevance to the given query. Only positively rated responses
        are retained.

        :param query: The user query used to assess community relevance.
        :param context: A list of community summary texts to evaluate.
        :return: A list of structured responses with fields such as ``response`` and ``rating``.
        """
        ...

    async def a_query(self, query: str) -> str | pydantic.BaseModel:
        """
        Execute a full global retrieval-augmented generation query.

        - Retrieves all community-level insights.
        - Generates a final global answer.

        :param query: The natural language query from the user.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        ...