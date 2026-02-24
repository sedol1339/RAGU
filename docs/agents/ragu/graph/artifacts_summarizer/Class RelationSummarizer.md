# Class RelationSummarizer (defined in ragu/graph/artifacts_summarizer.py at lines 221-362)

class RelationSummarizer(ragu.common.base.RaguGenerativeModule):
    """
    Summarizes and merges textual descriptions of entities and relations.
    extracted from documents.

    The class groups duplicated entities and relations by their identifiers
    (e.g., ``entity_name``, ``entity_type`` for entities, and
    ``subject_id``, ``object_id`` for relations), merges their attributes,
    and optionally generates concise descriptions through an LLM.

    :param client: LLM client used for summarization. Required if
                   ``use_llm_summarization=True``.
    :param use_llm_summarization: Whether to perform description summarization
                                  with a language model.
    :param language: Target language for summarization (e.g., ``"russian"`` or ``"english"``).
    :raises ValueError: If ``use_llm_summarization=True`` but no client is provided.
    """
    ...

    async def run(self, relations: typing.List[ragu.graph.types.Relation], **kwargs) -> typing.List[ragu.graph.types.Relation]:
        """
        Execute the full artifact summarization pipeline.

        The pipeline performs the following steps:

        1. Group duplicated relations into aggregated dataframes.
        2. Summarize merged entity and relation descriptions if enabled.
        3. Return the updated lists of :class:`Entity` and :class:`Relation` objects.

        :param relations: List of extracted relations to summarize or merge.
        :return: A tuple ``(entities, relations)`` containing updated objects.
        """
        ...

    async def summarize_relations(self, grouped_relations_df: pd.DataFrame) -> typing.List[ragu.graph.types.Relation]:
        """
        Summarize merged relation descriptions.

        Relations with identical pairs ``(subject_id, object_id)`` are combined
        into a single entry. If duplicates exist and LLM summarization is enabled,
        their descriptions are merged using the ``relation_summarizer`` prompt.

        :param grouped_relations_df: DataFrame containing grouped relation data with
                                     a ``duplicate_count`` column.
        :return: A list of summarized :class:`Relation` objects.
        """
        ...

    @staticmethod
    def group_relations(relations: typing.List[ragu.graph.types.Relation]) -> pd.DataFrame:
        """
        Group relations by (subject_id, object_id) and merge their fields.

        :param relations: List of Relation objects.
        :return: Aggregated relations as a pandas DataFrame.
        """
        ...