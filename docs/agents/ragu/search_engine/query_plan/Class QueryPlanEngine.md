# Class QueryPlanEngine (defined in ragu/search_engine/query_plan.py at lines 13-140)

class QueryPlanEngine(ragu.search_engine.base_engine.BaseEngine):
    """
    Query planning engine that decomposes complex queries into a DAG of subqueries
    and executes them in topological order.

    Pipeline:
      1. Decompose query -> list[SubQuery] (DAG)
      2. Topological sort
      3. For each subquery:
         - rewrite using dependency answers (if needed)
         - execute with underlying engine
         - store answer in context
      4. Return answer of the last subquery
    """
    ...

    async def process_query(self, query: str) -> typing.List[ragu.common.prompts.default_models.SubQuery]:
        """
        Decompose a complex query into atomic subqueries with dependencies.

        Uses an LLM to analyze the input query and break it down into minimal,
        independent subqueries. Each subquery is assigned a unique ID and may
        declare dependencies on other subqueries that must be resolved first.

        :param query: Complex natural-language query to decompose.
        :return: List of SubQuery objects forming a DAG.
        """
        ...

    async def a_query(self, query: str) -> str | pydantic.BaseModel:
        """
        Execute a complex query using the plan-and-execute pipeline.

        This method:
        1. Decompose the query into subqueries with dependencies.
        2. Sort subqueries in topological order.
        3. Rewrite subquery based on previous context and answer the query.
        4. Return the final answer from the last subquery

        Dependent subqueries are automatically rewritten to be self-contained
        by injecting answers from their prerequisite subqueries.

        :param query: The complex natural-language query to answer.
        :return: Pydantic model instance when a response schema is set, otherwise a plain string.
        :rtype: str | BaseModel
        """
        ...

    async def a_search(self, query, *args, **kwargs):
        """
        Perform a search using the underlying engine.

        :param query: The search query.
        :param args: Additional positional arguments passed to the underlying engine.
        :param kwargs: Additional keyword arguments passed to the underlying engine.
        :return: Search results from the underlying engine.
        """
        ...