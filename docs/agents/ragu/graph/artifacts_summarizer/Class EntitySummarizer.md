# Class EntitySummarizer (defined in ragu/graph/artifacts_summarizer.py at lines 21-219)

class EntitySummarizer(ragu.common.base.RaguGenerativeModule):
    """
    Summarizes and merges textual descriptions of duplicate entities.

    Entities are grouped by ``(entity_name, entity_type)``, merged, and optionally
    summarized with an LLM. When enabled, large description sets can be clustered
    before summarization to reduce prompt size.
    """
    ...

    async def run(self, entities: typing.List[ragu.graph.types.Entity]) -> typing.List[ragu.graph.types.Entity]:
        """
        Execute the full artifact summarization pipeline.

        Steps:
          1. Group duplicated entities by (entity_name, entity_type),
          2. Optionally cluster large description sets and summarize cluster-wise,
          3. Optionally summarize entities with many duplicates via LLM,
          4. Return updated list of Entity objects.

        :param entities: List of extracted entities.
        :return: Summarized/deduplicated entities list.
        """
        ...

    async def summarize_entities(self, grouped_entities_df: pd.DataFrame) -> typing.List[ragu.graph.types.Entity]:
        """
        Summarize merged entity descriptions.

        Entities with identical ``entity_name`` and ``entity_type`` are grouped
        into a single record. .

        :param grouped_entities_df: DataFrame containing grouped entity data with
                                    a ``duplicate_count`` column.
        :return: A list of summarized :class:`Entity` objects.
        """
        ...

    @staticmethod
    def group_entities(entities: typing.List[ragu.graph.types.Entity]) -> pd.DataFrame:
        """
        Group entities by ``entity_name`` and ``entity_type`` and aggregate their
        fields into combined records.

        :param entities: List of :class:`Entity` objects to group.
        :return: Aggregated entities as a :class:`pandas.DataFrame`.
        """
        ...