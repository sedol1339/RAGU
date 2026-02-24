# Class BaseEngine (defined in ragu/search_engine/base_engine.py at lines 12-72)

class BaseEngine(ragu.common.base.RaguGenerativeModule, abc.ABC):
    """
    Base interface for RAGU query/search engines.

    Concrete engines implement retrieval (a_search method) and answer generation
    (a_query method) on top of a knowledge graph.
    """
    ...

    @abc.abstractmethod
    async def a_search(self, query, *args, **kwargs) -> ragu.search_engine.types.NaiveSearchResult | ragu.search_engine.types.LocalSearchResult | ragu.common.prompts.default_models.GlobalSearchContextModel:
        """
        Retrieve context relevant to a query.

        :param query: Input query string.
        :return: Engine-specific retrieval result payload.
        """
        ...

    @abc.abstractmethod
    async def a_query(self, query: str) -> str | pydantic.BaseModel:
        """
        Execute full query flow and return answer.

        :param query: Input query string.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        ...

    async def query(self, query: str) -> str | pydantic.BaseModel:
        """
        Synchronous wrapper for ``a_query``.

        :param query: Input query string.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        ...

    async def search(self, query, *args, **kwargs) -> ragu.search_engine.types.NaiveSearchResult | ragu.search_engine.types.LocalSearchResult | ragu.common.prompts.default_models.GlobalSearchContextModel:
        """
        Synchronous wrapper for ``a_search``.

        :param query: Input query string.
        :return: Engine-specific retrieval result payload.
        """
        ...